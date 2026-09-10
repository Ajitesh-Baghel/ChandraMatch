import threading
import uuid
from datetime import datetime, timezone

_lock = threading.Lock()
_jobs = {}


def create_job(job_dir, source_sensor, reference_sensor, job_id=None):
    if job_id is None:
        job_id = uuid.uuid4().hex[:12]

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "job_dir": str(job_dir),
            "detected_source_sensor": source_sensor,
            "detected_reference_sensor": reference_sensor,
            "verdict": None,
            "verdict_reasons": [],
            "stage": None,
            "error": None,
            "files": {},
            "metrics": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    return job_id


def update_job(job_id, **fields):
    with _lock:
        if job_id not in _jobs:
            raise KeyError(job_id)

        _jobs[job_id].update(fields)


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)

        return dict(job) if job is not None else None
