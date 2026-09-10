"""
Stage 0a: raw archive ingestion. Currently a direct move of
app/isis_ingestion.py (see the bottom of that file, now a thin re-export for
backward compatibility) with no behavior change -- the ISIS chain, WSL
plumbing, and the LRO-NAC-only scope decision are all unchanged and already
confirmed working end-to-end on real data.
"""

import re
import shlex
import subprocess
from pathlib import Path

CONDA_ACTIVATE = (
    "source /home/ajitesh/miniforge3/etc/profile.d/conda.sh && conda activate isis10.0.0"
)

# Generous: cam2map alone took ~11 minutes for one NAC framelet in testing,
# dominated by I/O across the WSL<->Windows mount, not CPU.
ISIS_TIMEOUT_SECONDS = 1800

# Only LRO NAC has an automated raw-ingestion path so far -- OHRC (custom
# Delaunay/LinearNDInterpolator geometry, see src/ingestion/ohrc_geometry.py),
# SELENE/Kaguya, and IIRS (band-averaging + LOC-cube georeferencing) raw
# ingestion are explicitly out of scope for this pass, not silently guessed
# at. See docs/AGENT_STATE.md / project memory for the reconciliation history.
SUPPORTED_SENSORS = {"LRO NAC"}

# Every TMC-2 canonical raster in this project uses South Polar Stereographic
# centered exactly at the pole (CenterLatitude=-90, CenterLongitude=0). ISIS's
# cam2map defaults to Sinusoidal, which is badly distorted for the near-polar
# latitudes (~-85 deg) this project's data sits at -- confirmed by real
# testing: a Sinusoidal NAC footprint reprojected into Polar Stereographic
# produced a wildly inflated bounding box, large enough to trip Stage 0's
# dense-search memory-safety check. Projecting NAC directly into the SAME
# polar stereographic family as the reference avoids that distortion.
ISIS_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "app" / "isis_templates"
SOUTH_POLAR_MAP_TEMPLATE = ISIS_TEMPLATES_DIR / "south_polar_stereographic.map"


def windows_path_to_wsl(path):
    """
    E:\\foo\\bar -> /mnt/e/foo/bar. Confirmed on this machine (not assumed)
    that /mnt/<drive>/... is the live WSL mount for Windows drives.
    """

    path = Path(path).resolve()
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix()[len(path.drive) :].lstrip("/")

    return f"/mnt/{drive}/{rest}"


def run_isis_lro_nac(img_path, work_dir):
    """
    Runs the LRO NAC ISIS chain via a single WSL subprocess call. Confirmed
    working end-to-end on real data in this environment: lronac2isis (~6s) ->
    spiceinit web=yes (fetches only the small time-scoped kernel slice for
    this exact observation, never the 759GB full mission archive) ->
    lronaccal (~13s) -> lronacecho (~6s) -> cam2map (~11min, dominated by
    WSL<->Windows mount I/O, not CPU). Output uses South Polar Stereographic
    (see module docstring above). GDAL's ISIS3 driver reads the resulting
    .cub directly with correct CRS/transform/nodata already set -- no extra
    conversion step needed.

    Returns the Windows path to the final mapped .cub file.
    """

    img_path = Path(img_path).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem

    # Defense in depth: the upload endpoint already sanitizes filenames, but
    # this function embeds `stem` directly into a shell script string
    # (executed via wsl.exe) and into ISIS's own key=value argument syntax
    # (where a stray "=" would break parsing regardless of shell quoting) --
    # never trust an upstream caller to have done this, reject anything
    # unexpected outright rather than trying to escape it.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", stem):
        raise ValueError(
            f"Unsafe or unexpected filename stem for ISIS processing: {stem!r}"
        )

    wsl_img_path = windows_path_to_wsl(img_path)
    wsl_work_dir = windows_path_to_wsl(work_dir)
    wsl_map_template = windows_path_to_wsl(SOUTH_POLAR_MAP_TEMPLATE)

    quoted_img_path = shlex.quote(wsl_img_path)
    quoted_work_dir = shlex.quote(wsl_work_dir)
    quoted_map_template = shlex.quote(wsl_map_template)

    script = f"""#!/bin/bash
set -e
{CONDA_ACTIVATE}
cd {quoted_work_dir}
lronac2isis from={quoted_img_path} to={stem}.cub
spiceinit from={stem}.cub web=yes
lronaccal from={stem}.cub to={stem}.cal.cub
lronacecho from={stem}.cal.cub to={stem}.echo.cub
cam2map from={stem}.echo.cub to={stem}.map.cub pixres=mpp resolution=5.0 defaultrange=camera trim=yes map={quoted_map_template}
"""

    # Written to a real file and run under Linux's own `timeout`, rather than
    # passed as a `bash -lc` string with only Python's subprocess timeout as
    # the deadline -- that alone only kills the wsl.exe launcher on the
    # Windows side, not the actual ISIS process tree running inside the
    # persistent WSL instance.
    script_path = work_dir / f"{stem}_isis_run.sh"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    wsl_script_path = shlex.quote(windows_path_to_wsl(script_path))

    result = subprocess.run(
        [
            "wsl.exe",
            "-e",
            "timeout",
            f"{ISIS_TIMEOUT_SECONDS}s",
            "bash",
            wsl_script_path,
        ],
        capture_output=True,
        text=True,
        timeout=ISIS_TIMEOUT_SECONDS + 60,
    )

    log_path = work_dir / f"{stem}_isis_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- stdout ---\n")
        f.write(result.stdout)
        f.write("\n--- stderr ---\n")
        f.write(result.stderr)

    output_cub = work_dir / f"{stem}.map.cub"

    if result.returncode != 0 or not output_cub.exists():
        raise RuntimeError(
            f"ISIS LRO NAC processing failed (exit {result.returncode}). "
            f"See {log_path} for the full log."
        )

    return output_cub


def needs_isis(file_path, sensor_label):
    suffix = Path(file_path).suffix.lower()
    return suffix == ".img" and sensor_label in SUPPORTED_SENSORS


def is_unsupported_raw_format(file_path, sensor_label):
    """
    True if this is a raw .img upload for a sensor we don't have an
    automated ingestion path for yet (everything except LRO NAC).
    """

    suffix = Path(file_path).suffix.lower()

    return suffix == ".img" and sensor_label not in SUPPORTED_SENSORS


def ingest(path, sensor_label, work_dir):
    """
    Stage 0a entry point used by run.py::register(). Returns the path to a
    calibrated, map-projected GeoTIFF/cube ready for canonicalize.py, running
    ISIS first if (and only if) this is a raw .IMG this project has an
    automated path for; otherwise returns `path` unchanged (already a
    GeoTIFF, or a raw format explicitly unsupported -- the caller is
    responsible for treating `is_unsupported_raw_format` as a hard FAIL
    before calling this).
    """

    if needs_isis(path, sensor_label):
        return run_isis_lro_nac(path, work_dir)

    return Path(path)
