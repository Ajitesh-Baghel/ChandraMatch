from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

DOCS = ROOT / "docs" / "images"
DOCS.mkdir(parents=True, exist_ok=True)


assets = {

    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_001"
    / "pair_comparison.png":

        "01_tmc_pair001.png",


    ROOT
    / "results"
    / "pair_001"
    / "sift"
    / "ransac_inliers.png":

        "02_sift_correspondences.png",


    ROOT
    / "results"
    / "pair_001"
    / "lightglue"
    / "ransac_inliers.png":

        "03_lightglue_correspondences.png",


    ROOT
    / "results"
    / "pair_001"
    / "loftr"
    / "ransac_inliers.png":

        "04_loftr_correspondences.png",


    ROOT
    / "results"
    / "pair_001"
    / "sift"
    / "overlay.png":

        "05_registered_overlay.png",
}


for source, target_name in assets.items():

    if not source.exists():

        print(
            "Missing:",
            source
        )

        continue

    destination = (
        DOCS / target_name
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        "Copied:",
        destination
    )