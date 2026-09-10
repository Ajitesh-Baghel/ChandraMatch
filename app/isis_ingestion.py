import re
import shlex
import subprocess
from pathlib import Path

CONDA_ACTIVATE = (
    "source /home/ajitesh/miniforge3/etc/profile.d/conda.sh && conda activate isis10.0.0"
)

# Generous: cam2map alone took ~11 minutes for one NAC framelet in
# testing, dominated by I/O across the WSL<->Windows mount, not CPU.
ISIS_TIMEOUT_SECONDS = 1800

SUPPORTED_SENSORS = {"LRO NAC"}

# Every TMC-2 canonical raster in this project uses South Polar
# Stereographic centered exactly at the pole (CenterLatitude=-90,
# CenterLongitude=0). ISIS's cam2map defaults to Sinusoidal, which is
# badly distorted for the near-polar latitudes (~-85 deg) this
# project's data sits at -- confirmed by real testing: a Sinusoidal NAC
# footprint reprojected into Polar Stereographic produced a wildly
# inflated bounding box (13827x6058, ~84M px) versus the true physical
# footprint, large enough to trip Stage 0's dense-search memory-safety
# check. Projecting NAC directly into the SAME polar stereographic
# family as the reference avoids that distortion entirely.
ISIS_TEMPLATES_DIR = Path(__file__).resolve().parent / "isis_templates"
SOUTH_POLAR_MAP_TEMPLATE = ISIS_TEMPLATES_DIR / "south_polar_stereographic.map"


def windows_path_to_wsl(path):
    """
    E:\\foo\\bar -> /mnt/e/foo/bar. Confirmed on this machine (not
    assumed) that /mnt/<drive>/... is the live WSL mount for Windows
    drives.
    """

    path = Path(path).resolve()
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix()[len(path.drive) :].lstrip("/")

    return f"/mnt/{drive}/{rest}"


def run_isis_lro_nac(img_path, work_dir):
    """
    Runs the LRO NAC ISIS chain via a single WSL subprocess call.
    Confirmed working end-to-end on real data in this environment (not
    assumed from documentation) -- each step's approximate runtime, also
    confirmed by direct testing:
      lronac2isis   ~6s     raw EDR -> ISIS cube
      spiceinit     ~varies fetches ONLY the small time-scoped kernel
                            slice for this exact observation via NASA's
                            web service (web=yes) -- confirmed this is
                            NOT the full mission archive, which is 759GB
                            and must never be bulk-downloaded locally
      lronaccal     ~13s    radiometric calibration
      lronacecho    ~6s     echo/dark correction
      cam2map       ~11min  map-projection at 5 m/px (the slow step,
                            mostly I/O across the WSL<->Windows mount)

    Output uses South Polar Stereographic (via SOUTH_POLAR_MAP_TEMPLATE),
    matching every TMC-2 canonical raster in this project -- NOT ISIS's
    own default (Sinusoidal), which real testing showed is badly
    distorted for this project's near-polar data and produces a wildly
    inflated bounding box after reprojection (confirmed: one real test
    case went from a normal-sized footprint to 13827x6058, ~84M px,
    large enough to trip Stage 0's memory-safety check). Confirmed
    GDAL's ISIS3 driver reads the resulting .cub directly with correct
    CRS/transform/nodata already set, so no extra conversion step
    (isis2std/gdal_translate) is needed.

    Returns the Windows path to the final mapped .cub file.
    """

    img_path = Path(img_path).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem

    # Defense in depth: the upload endpoint already sanitizes filenames,
    # but this function embeds `stem` directly into a shell script string
    # (executed via `wsl.exe -e bash -lc`) and into ISIS's own `key=value`
    # argument syntax (where a stray "=" would break parsing regardless of
    # shell quoting) -- never trust an upstream caller to have done this,
    # reject anything unexpected outright rather than trying to escape it.
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

    # Written to a real file and run under Linux's own `timeout`, rather
    # than passed as a `bash -lc` string with only Python's subprocess
    # timeout as the deadline -- that alone only kills the wsl.exe
    # launcher on the Windows side, not the actual ISIS process tree
    # running inside the persistent WSL instance (confirmed: WSL2 keeps
    # the VM and its processes running independently of the launcher).
    # `timeout` runs natively inside WSL and can actually kill its own
    # child process group when the deadline hits.
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
        # Generous safety-net margin on top of the WSL-side `timeout`,
        # which is the one actually expected to fire first.
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
    automated ingestion path for yet (everything except LRO NAC, per
    the explicit scoping decision to build LRO NAC first and treat
    OHRC/Kaguya/IIRS raw ingestion as documented future work rather than
    guess at it).
    """

    suffix = Path(file_path).suffix.lower()

    return suffix == ".img" and sensor_label not in SUPPORTED_SENSORS
