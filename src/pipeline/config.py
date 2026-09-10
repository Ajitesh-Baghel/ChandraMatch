"""
Single, versioned config object covering every stage's shared thresholds.

Design note: this wraps the three existing per-stage JSON files
(configs/stage{0,1,2}_*.json) rather than merging them into one physical
file. Those three files are already the single shared source of truth for
every pair (rule: "one shared config per stage, replayed unmodified across
every pair" -- already true today) and are referenced by name from existing,
validated scripts and docs; merging them into one file right now would be
pure churn with no behavioral benefit and would multiply the regression risk
this pass is trying to minimize. PipelineConfig is the ONE in-memory object
every stage function is handed explicitly (never a global, never a
per-pair copy) -- that is the actual unification this satisfies.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE0_PATH = ROOT / "configs" / "stage0_coarse_search.json"
DEFAULT_STAGE1_PATH = ROOT / "configs" / "stage1_local_control_points.json"
DEFAULT_STAGE2_PATH = ROOT / "configs" / "stage2_registration.json"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.pop("notes", None)

    return data


@dataclass
class PipelineConfig:
    stage0: dict
    stage1: dict
    stage2: dict
    version: str = CONFIG_VERSION

    # Optional, per-call sensor overrides -- NOT auto-detection tuning. Set
    # by a caller (CLI flag, or the web app after the user confirms a
    # low-confidence auto-detection) when the label is already known;
    # register() still runs sensor_id.detect_sensor() when these are None,
    # per the "only file paths and config as inputs" contract in the
    # unification brief. Never used to force a *wrong* label -- only to
    # skip re-detecting one the caller already confirmed.
    source_sensor_override: str | None = field(default=None)
    reference_sensor_override: str | None = field(default=None)


def load_pipeline_config(
    stage0_path=DEFAULT_STAGE0_PATH,
    stage1_path=DEFAULT_STAGE1_PATH,
    stage2_path=DEFAULT_STAGE2_PATH,
    source_sensor_override=None,
    reference_sensor_override=None,
):
    return PipelineConfig(
        stage0=_load_json(stage0_path),
        stage1=_load_json(stage1_path),
        stage2=_load_json(stage2_path),
        source_sensor_override=source_sensor_override,
        reference_sensor_override=reference_sensor_override,
    )
