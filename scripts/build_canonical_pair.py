import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonicalization.build_canonical_pair import build_canonical_pair


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generic canonicalization: reproject a source (moving) raster onto "
            "a common grid derived from a reference (fixed) raster, recover the "
            "reference's true footprint, and build science/matcher masks -- "
            "the same pattern used for every real pair in this project, "
            "generalized to work on any new source/reference pair without a "
            "bespoke script."
        )
    )
    parser.add_argument("--source", required=True, help="Path to the source (moving) raster")
    parser.add_argument("--reference", required=True, help="Path to the reference (fixed) raster")
    parser.add_argument("--output-dir", required=True, help="Directory for canonical outputs")
    parser.add_argument("--pair-id", required=True, help="Identifier used in filenames/metadata")
    parser.add_argument("--source-sensor", default="unknown")
    parser.add_argument("--source-product", default="unknown")
    parser.add_argument("--reference-sensor", default="unknown")
    parser.add_argument("--reference-product", default="unknown")
    parser.add_argument(
        "--target-resolution-m",
        type=float,
        default=None,
        help="Override common grid resolution (default: reference's native resolution)",
    )
    parser.add_argument("--mask-erosion-px", type=int, default=5)

    args = parser.parse_args()

    print("=" * 78)
    print(f"CANONICALIZE -- {args.pair_id}")
    print("=" * 78)

    paths, metadata = build_canonical_pair(
        source_path=args.source,
        reference_path=args.reference,
        output_dir=args.output_dir,
        pair_id=args.pair_id,
        source_sensor=args.source_sensor,
        source_product=args.source_product,
        reference_sensor=args.reference_sensor,
        reference_product=args.reference_product,
        target_resolution_m=args.target_resolution_m,
        mask_erosion_px=args.mask_erosion_px,
    )

    print("\nCanonical shape:", metadata["width"], "x", metadata["height"])
    print("Resolution:", metadata["canonical_resolution_m"], "m/px")
    print("Science-valid pixels:", f"{metadata['science_valid_pixels']:,}")
    print("Matcher-valid pixels:", f"{metadata['matcher_pixels']:,}", f"({metadata['matcher_ratio']:.4f})")
    print("Matcher bbox:", metadata["matcher_bbox"])

    print("\nOutputs:")
    for name, path in paths.items():
        print(f"  {name:<20} {path}")

    print(f"\nCANONICALIZATION COMPLETE -- {args.pair_id}")


if __name__ == "__main__":
    main()
