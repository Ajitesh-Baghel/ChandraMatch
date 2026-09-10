import csv

import cv2
import numpy as np
import rasterio


def register_source(source_data, source_valid, reference_valid, affine_matrix):
    """
    Coverage-weighted warp, ported from v1's
    scripts/finalize_pair004_product.py::register_source (already
    pair-agnostic there despite living in a per-pair script). Warping
    pixel values and a validity weight separately, then dividing, avoids
    nodata contamination bleeding into the registered footprint's edges
    -- a plain `cv2.warpAffine` on raw values does not have this
    property when invalid source pixels sit near the boundary.
    """

    height, width = source_data.shape

    source_clean = np.where(source_valid, source_data, 0.0).astype(np.float32)
    source_weight = source_valid.astype(np.float32)

    warped_value = cv2.warpAffine(
        source_clean,
        affine_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_weight = cv2.warpAffine(
        source_weight,
        affine_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    registered = np.zeros((height, width), dtype=np.float32)

    valid = (warped_weight > 0.50) & reference_valid

    registered[valid] = warped_value[valid] / np.maximum(warped_weight[valid], 1e-6)

    return registered, valid


def choose_nodata(source_nodata):
    if source_nodata is not None and np.isfinite(source_nodata):
        value = np.float32(source_nodata)

        if np.isfinite(value):
            return float(value)

    return float(np.float32(-3.4028235e38))


def write_registered_geotiff(path, registered, valid, reference_profile, source_nodata):
    nodata = choose_nodata(source_nodata)

    output = registered.copy()
    output[~valid] = nodata

    height, width = registered.shape

    profile = reference_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    # GDAL's tiled-TIFF blocks must be multiples of 16 -- for small
    # rasters (e.g. Pair002's 832x990 canonical crop) that constraint
    # can conflict with the image's own dimensions, so only tile large
    # enough images and let GDAL use plain strips otherwise.
    use_tiling = height >= 256 and width >= 256

    profile.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 3,
            "tiled": use_tiling,
            "BIGTIFF": "IF_SAFER",
        }
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output.astype(np.float32), 1)
        dst.write_mask(valid.astype(np.uint8) * 255)

    return nodata


def robust_uint8(image, valid):
    out = np.zeros(image.shape, dtype=np.uint8)

    values = image[valid]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return out

    low, high = np.percentile(values, [2.0, 98.0])

    if high <= low:
        high = low + 1.0

    normalized = (image.astype(np.float32) - float(low)) / float(high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    out[valid] = np.round(normalized[valid] * 255.0).astype(np.uint8)

    return out


def save_overlay(registered_preview, reference_preview, valid, path):
    reg = cv2.cvtColor(registered_preview, cv2.COLOR_GRAY2BGR)
    ref = cv2.cvtColor(reference_preview, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(reg, 0.5, ref, 0.5, 0.0)
    overlay[~valid] = 0

    cv2.imwrite(str(path), overlay)


def save_checkerboard(registered_preview, reference_preview, valid, path, block=128):
    height, width = registered_preview.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            y1 = min(height, y0 + block)
            x1 = min(width, x0 + block)

            if (x0 // block + y0 // block) % 2 == 0:
                result[y0:y1, x0:x1] = registered_preview[y0:y1, x0:x1]
            else:
                result[y0:y1, x0:x1] = reference_preview[y0:y1, x0:x1]

    result[~valid] = 0

    cv2.imwrite(str(path), result)


def write_correspondence_csv(points, path):
    fields = [
        "reference_x",
        "reference_y",
        "source_x",
        "source_y",
        "residual_dy",
        "residual_dx",
        "residual_magnitude_px",
        "peak_score",
        "ambiguity_ratio",
        "valid_fraction",
        "stage1_accepted",
        "fit_residual_px",
        "fit_outlier",
        "final_accepted",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p in points:
            writer.writerow({field: p.get(field) for field in fields})
