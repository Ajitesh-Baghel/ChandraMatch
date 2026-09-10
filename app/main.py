import re
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import job_store
from app.pipeline_runner import run_pipeline
from src.pipeline.sensor_id import detect_sensor

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "app" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ChandraMatch")


def sanitize_filename(filename):
    """
    Client-supplied filenames are untrusted. Strips everything except a
    safe character set so the result can never contain a path separator
    or traversal sequence, regardless of OS -- a raw filename was
    previously used directly to build an on-disk path (and, via the
    ISIS wrapper's shell script, was also exploitable for command
    injection), so this is a mandatory boundary, not a convenience.
    """

    name = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "upload")
    name = name.lstrip(".")

    return name or "upload"


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    source_sensor: str = Form(default=""),
    reference_sensor: str = Form(default=""),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    source_filename = sanitize_filename(source.filename)
    reference_filename = sanitize_filename(reference.filename)

    source_path = uploads_dir / source_filename
    reference_path = uploads_dir / reference_filename

    with open(source_path, "wb") as f:
        shutil.copyfileobj(source.file, f)

    with open(reference_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    detected_source = detect_sensor(source_path, source_filename)
    detected_reference = detect_sensor(reference_path, reference_filename)

    job_store.create_job(
        job_dir=job_dir,
        source_sensor=detected_source,
        reference_sensor=detected_reference,
        job_id=job_id,
    )

    # A user-provided override is passed through as-is (trusted). Otherwise
    # pass None and let register() run its OWN Stage -1 detection (same
    # sensor_id.detect_sensor logic) rather than the app pre-emptively
    # locking in a possibly-"unknown" label as if it were confirmed --
    # register() already turns an unknown sensor into an honest AMBIGUOUS
    # verdict asking for manual confirmation, which is the exact behavior
    # wanted here; duplicating that decision in the app would risk drifting
    # out of sync with it.
    background_tasks.add_task(
        run_pipeline,
        job_id,
        source_path,
        reference_path,
        job_dir,
        source_sensor.strip() or None,
        reference_sensor.strip() or None,
    )

    return {
        "job_id": job_id,
        "detected_source_sensor": detected_source,
        "detected_reference_sensor": detected_reference,
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    job.pop("job_dir", None)

    return job


@app.get("/api/jobs/{job_id}/files/{name}")
async def get_job_file(job_id: str, name: str):
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    path = job.get("files", {}).get(name)

    if path is None:
        raise HTTPException(status_code=404, detail=f"No such file for this job: {name}")

    return FileResponse(path)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
