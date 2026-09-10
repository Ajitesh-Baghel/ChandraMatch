import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds, intersection
from scipy import ndimage


def full_pixel_window(bounds, transform):
    """
    Integer pixel window on `transform`'s grid that fully covers
    `bounds`, rounded outward so no partial pixel is clipped.
    """

    w = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, transform=transform)

    c0 = math.floor(w.col_off)
    r0 = math.floor(w.row_off)
    c1 = math.ceil(w.col_off + w.width)
    r1 = math.ceil(w.row_off + w.height)

    return Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)


def recover_positive_support_footprint(positive_mask, close_kernel_px=21):
    """
    Recovers a raster's true observation footprint from a "value is
    positive" seed mask, via morphological closing (to bridge small
    internal gaps/noise) + largest external contour + convex hull.

    Needed because many lunar products (TMC-2 confirmed) don't set a
    real `nodata` value -- zero outside the observed swath and zero
    inside it (real shadow) are indistinguishable by value alone, so
    "value > 0" is only a seed for finding the swath shape, never the
    final science-validity definition on its own (a zero pixel INSIDE
    the recovered hull is still footprint-valid; it's just excluded
    from matcher/feature-extraction support elsewhere).
    """

    seed = positive_mask.astype(np.uint8) * 255

    kernel = np.ones((close_kernel_px, close_kernel_px), np.uint8)
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("Could not recover a geometric footprint from the positive-support seed.")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)

    footprint = np.zeros_like(seed)
    cv2.fillConvexPoly(footprint, hull, 255)

    return footprint > 0


def write_tif(path, array, transform, crs, nodata=None, dtype=None):
    if dtype is None:
        dtype = array.dtype

    height, width = array.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    if height >= 256 and width >= 256:
        profile["tiled"] = True

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def robust_preview_uint8(array, valid):
    out = np.zeros(array.shape, dtype=np.uint8)

    values = array[valid & np.isfinite(array)]

    if values.size == 0:
        return out

    low, high = np.percentile(values, [2, 98])

    if high <= low:
        high = low + 1.0

    scaled = np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    out = (scaled * 255).astype(np.uint8)
    out[~valid] = 0

    return out


def _rescale_transform(transform, target_resolution_m, source_resolution_m):
    scale = target_resolution_m / source_resolution_m

    return rasterio.Affine(
        transform.a * scale,
        transform.b,
        transform.c,
        transform.d,
        transform.e * scale,
        transform.f,
    )


def build_canonical_pair(
    source_path,
    reference_path,
    output_dir,
    pair_id,
    source_sensor="unknown",
    source_product="unknown",
    reference_sensor="unknown",
    reference_product="unknown",
    target_resolution_m=None,
    mask_erosion_px=5,
    footprint_close_kernel_px=21,
    source_resampling=Resampling.bilinear,
):
    """
    Generic canonicalization: reproject SOURCE (the moving image, per
    the PS's own terminology) onto a common grid derived from
    REFERENCE (the fixed image), crop to source's extent, recover
    reference's true observation footprint (robust to unreliable/
    missing nodata), build science-valid vs matcher-valid masks, and
    write canonical rasters + masks + previews + metadata.

    This generalizes the pattern independently duplicated across
    src/overlap/build_pair_00{1,6,7}.py, scripts/build_pair004.py/
    build_pair005.py, and scripts/build_pair_00{2,3}_*.py into one
    reusable function -- a new pair no longer needs its own bespoke
    canonicalization script.

    `target_resolution_m`: None (default) uses reference's own native
    pixel size, matching what Pair004/006/007/002 all did. Pass an
    explicit value to build a common grid at a DIFFERENT resolution
    (e.g. Pair003/IIRS's choice of 63.37 m to avoid upsampling IIRS's
    coarser native data onto TMC-2's 5 m grid) -- same origin/CRS as
    reference, just a different pixel size, still cropped to source's
    extent.

    Returns a dict of output paths and the metadata dict (also written
    to `<output_dir>/<pair_id>_canonical_metadata.json`).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref_ds, rasterio.open(source_path) as src_ds:
        reference_resolution_m = float(abs(ref_ds.transform.a))

        base_transform = ref_ds.transform

        if target_resolution_m is not None and not math.isclose(
            target_resolution_m, reference_resolution_m, rel_tol=1e-6
        ):
            base_transform = _rescale_transform(
                ref_ds.transform, target_resolution_m, reference_resolution_m
            )

        # Bug found via real testing (a raw ISIS LRO NAC cube in
        # Sinusoidal projection against TMC-2's Polar Stereographic
        # reference): every prior test happened to use source/reference
        # files that already shared a compatible CRS, so using
        # src_ds.bounds directly against the reference's transform never
        # surfaced the missing reprojection step. Must transform the
        # source's bounding box into the reference's CRS first whenever
        # the two actually differ.
        if src_ds.crs is not None and src_ds.crs != ref_ds.crs:
            left, bottom, right, top = transform_bounds(
                src_ds.crs, ref_ds.crs, *src_ds.bounds
            )
            source_bounds_in_ref_crs = rasterio.coords.BoundingBox(
                left, bottom, right, top
            )
        else:
            source_bounds_in_ref_crs = src_ds.bounds

        # Bug found via real testing (a raw LRO NAC framelet against
        # TMC-2): the target window was previously bounded ONLY by the
        # source's own extent, never intersected with the reference's
        # actual valid extent. A narrow, map-grid-unaligned swath's
        # axis-aligned bounding box can be far larger than its real
        # content (confirmed: one real case went from a genuine ~4992x884
        # overlap to a 13670x5297, ~72M px canvas), which is both wasteful
        # and can trip Stage 0's memory-safety limit. Intersecting with
        # the reference's own extent bounds the canvas by whichever
        # extent is actually smaller, instead of only ever the source's.
        source_window = full_pixel_window(source_bounds_in_ref_crs, base_transform)
        reference_window = full_pixel_window(ref_ds.bounds, base_transform)

        try:
            target_window = intersection(source_window, reference_window)
        except WindowError as exc:
            raise RuntimeError(
                f"{pair_id}: source and reference bounds do not overlap "
                f"on the reference grid ({exc})."
            ) from exc

        target_transform = rasterio.Affine(
            base_transform.a,
            base_transform.b,
            base_transform.c + target_window.col_off * base_transform.a,
            base_transform.d,
            base_transform.e,
            base_transform.f + target_window.row_off * base_transform.e,
        )

        width = int(target_window.width)
        height = int(target_window.height)

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"{pair_id}: source and reference bounds do not overlap "
                "on the reference grid."
            )

        # --------------------------------------------------------
        # Reference on the target grid (native-resolution case: read
        # directly; custom-resolution case: reproject).
        # --------------------------------------------------------

        if target_resolution_m is None:
            reference_arr = ref_ds.read(1, window=target_window, boundless=True, fill_value=0)
        else:
            reference_arr = np.zeros((height, width), dtype=ref_ds.dtypes[0])

            reproject(
                source=rasterio.band(ref_ds, 1),
                destination=reference_arr,
                src_transform=ref_ds.transform,
                src_crs=ref_ds.crs,
                dst_transform=target_transform,
                dst_crs=ref_ds.crs,
                resampling=Resampling.bilinear,
            )

        reference_finite = np.isfinite(reference_arr)
        reference_positive = reference_finite & (reference_arr > 0)

        # --------------------------------------------------------
        # Reproject source onto the same target grid.
        # --------------------------------------------------------

        source_on_grid = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src_ds, 1),
            destination=source_on_grid,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_ds.nodata,
            dst_transform=target_transform,
            dst_crs=ref_ds.crs,
            dst_nodata=np.nan,
            resampling=source_resampling,
        )

        source_native = src_ds.read(1)
        source_native_valid = np.isfinite(source_native)

        if src_ds.nodata is not None:
            source_native_valid &= source_native != src_ds.nodata

        source_valid_on_grid = np.zeros((height, width), dtype=np.uint8)

        reproject(
            source=source_native_valid.astype(np.uint8),
            destination=source_valid_on_grid,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=target_transform,
            dst_crs=ref_ds.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        source_valid = (source_valid_on_grid > 0) & np.isfinite(source_on_grid)

        # --------------------------------------------------------
        # Reference footprint: use real nodata/mask if the dataset
        # actually declares one reliably; otherwise recover it from
        # the positive-support seed (the TMC-2 nodata=None situation).
        # --------------------------------------------------------

        if ref_ds.nodata is not None:
            reference_footprint = reference_finite
        else:
            reference_footprint = recover_positive_support_footprint(
                reference_positive, footprint_close_kernel_px
            )

        # --------------------------------------------------------
        # Masks: science (footprint-based, keeps real zero/shadow) vs
        # matcher (positive-support only, eroded -- excluded from
        # feature extraction, not from science validity).
        # --------------------------------------------------------

        science_mask = source_valid & reference_footprint
        matcher_base = source_valid & reference_positive

        matcher_mask = (
            ndimage.binary_erosion(
                matcher_base,
                structure=np.ones((3, 3), dtype=bool),
                iterations=mask_erosion_px,
                border_value=0,
            )
            if mask_erosion_px > 0
            else matcher_base
        )

        if matcher_mask.sum() == 0:
            raise RuntimeError(f"{pair_id}: no common matcher support found.")

        ys, xs = np.where(matcher_mask)
        matcher_bbox = {
            "min_x": int(xs.min()),
            "max_x": int(xs.max()),
            "min_y": int(ys.min()),
            "max_y": int(ys.max()),
            "width": int(xs.max() - xs.min() + 1),
            "height": int(ys.max() - ys.min() + 1),
        }

        # --------------------------------------------------------
        # Write canonical rasters + masks
        # --------------------------------------------------------

        paths = {
            "source": output_dir / "source.tif",
            "reference": output_dir / "reference.tif",
            "science_mask": output_dir / "science_mask.tif",
            "matcher_mask": output_dir / "matcher_mask.tif",
            "reference_footprint": output_dir / "reference_footprint_mask.tif",
        }

        write_tif(paths["source"], source_on_grid, target_transform, ref_ds.crs, nodata=np.nan, dtype="float32")
        write_tif(paths["reference"], reference_arr, target_transform, ref_ds.crs, nodata=None, dtype=reference_arr.dtype)
        write_tif(paths["science_mask"], science_mask.astype(np.uint8), target_transform, ref_ds.crs, nodata=0, dtype="uint8")
        write_tif(paths["matcher_mask"], matcher_mask.astype(np.uint8), target_transform, ref_ds.crs, nodata=0, dtype="uint8")
        write_tif(paths["reference_footprint"], reference_footprint.astype(np.uint8), target_transform, ref_ds.crs, nodata=0, dtype="uint8")

        # --------------------------------------------------------
        # Previews
        # --------------------------------------------------------

        source_preview = robust_preview_uint8(source_on_grid, science_mask)
        reference_preview = robust_preview_uint8(
            reference_arr.astype(np.float32), science_mask & reference_positive
        )

        cv2.imwrite(str(output_dir / "source_preview.png"), source_preview)
        cv2.imwrite(str(output_dir / "reference_preview.png"), reference_preview)
        cv2.imwrite(str(output_dir / "matcher_mask_preview.png"), matcher_mask.astype(np.uint8) * 255)

        side_by_side = np.concatenate([reference_preview, source_preview], axis=1)
        cv2.imwrite(str(output_dir / "side_by_side_preview.png"), side_by_side)

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        metadata = {
            "pair_id": pair_id,
            "source_sensor": source_sensor,
            "source_product": source_product,
            "reference_sensor": reference_sensor,
            "reference_product": reference_product,
            "canonical_resolution_m": float(abs(target_transform.a)),
            "width": width,
            "height": height,
            "reference_positive_pixels": int(reference_positive.sum()),
            "reference_positive_ratio": float(reference_positive.mean()),
            "reference_footprint_pixels": int(reference_footprint.sum()),
            "reference_footprint_ratio": float(reference_footprint.mean()),
            "source_valid_pixels": int(source_valid.sum()),
            "source_valid_ratio": float(source_valid.mean()),
            "science_valid_pixels": int(science_mask.sum()),
            "science_valid_ratio": float(science_mask.mean()),
            "matcher_base_pixels": int(matcher_base.sum()),
            "matcher_base_ratio": float(matcher_base.mean()),
            "matcher_pixels": int(matcher_mask.sum()),
            "matcher_ratio": float(matcher_mask.mean()),
            "matcher_bbox": matcher_bbox,
            "transform": list(target_transform)[:6],
            "crs": str(ref_ds.crs),
            "notes": [
                "SOURCE was reprojected onto a grid derived from REFERENCE "
                "(native resolution unless target_resolution_m was given).",
                "REFERENCE positive support (`value > 0`) is a matcher-support "
                "seed, not the science-validity definition -- a real nodata "
                "value on the reference dataset is used directly when present; "
                "otherwise the footprint is recovered via morphological "
                "closing + convex hull so internal zero-valued pixels (e.g. "
                "real shadow) remain scientifically valid.",
                "Matcher mask uses common source/reference positive support "
                f"followed by {mask_erosion_px}px erosion.",
                "No image-registration transform has been applied at this stage.",
            ],
        }

        metadata_path = output_dir / f"{pair_id}_canonical_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        paths["metadata"] = metadata_path

        return paths, metadata
