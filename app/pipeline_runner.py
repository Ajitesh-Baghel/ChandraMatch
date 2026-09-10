from pathlib import Path

from app import job_store
from src.pipeline.config import load_pipeline_config
from src.pipeline.run import register


def run_pipeline(
    job_id,
    source_path,
    reference_path,
    job_dir,
    source_sensor_override,
    reference_sensor_override,
):
    """
    Thin adapter between the upload job model and the ONE real pipeline
    entry point, src.pipeline.run.register() -- the same function the CLI
    regression harness (scripts/regression_register_pair.py) calls for every
    validated pair. This replaces an earlier version of this file that
    hand-duplicated Stage 0/1/2 orchestration directly against src/coarse/*,
    which meant the app never received any of the unified-pipeline fixes
    (sub_pixel_achieved honesty, verification_tier, the anchor-confidence
    gate, the §6 export schema) -- register() is the single source of truth
    now, exactly as its own docstring already claimed.

    Empty-string overrides are treated as "no override" (auto-detect),
    matching load_pipeline_config's own contract.
    """

    job_dir = Path(job_dir)

    try:
        config = load_pipeline_config(
            source_sensor_override=source_sensor_override or None,
            reference_sensor_override=reference_sensor_override or None,
        )

        job_store.update_job(job_id, status="running", stage="register")

        result = register(source_path, reference_path, config, out_dir=job_dir)

        files = {name: str(path) for name, path in result.files.items()}

        job_store.update_job(
            job_id,
            status="done",
            stage="complete",
            verdict=result.verdict,
            verdict_reasons=result.verdict_reasons,
            metrics=result.metrics,
            files=files,
        )
    except Exception as exc:
        job_store.update_job(job_id, status="failed", error=repr(exc))
        raise
