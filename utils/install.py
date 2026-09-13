"""Install the JFER extension into an MTR checkout."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASE_FILES = {
    "setup.py": "5d50c987ba41f3d77307a511244ba57dbe4f2de4595c5510f248947a0e6cc7a2",
    "mtr/datasets/dataset.py": "bcf3cace97e640bab9aca3a3954c42b437950ecf80870d6fdfa1896ff61c2a32",
    "mtr/models/utils/common_layers.py": "89f5c772624e4866172428e17fa9dec6fa7f13121f406ee1272d1e14b544156c",
}

FILES = {
    "models/backbone/model.py": "mtr/models/model.py",
    "data/builders.py": "mtr/datasets/__init__.py",
    "models/backbone/common.py": "mtr/utils/common_utils.py",
    "models/backbone/attention.py": (
        "mtr/models/utils/transformer/multi_head_attention.py"
    ),
    "models/backbone/decoder_layer.py": (
        "mtr/models/utils/transformer/transformer_decoder_layer.py"
    ),
    "utils/training_runner.py": "tools/train.py",
    "evaluation/runner.py": "tools/test.py",
    "utils/training_utils.py": "tools/train_utils/train_utils.py",
    "evaluation/evaluation_utils.py": "tools/eval_utils/eval_utils.py",
    "evaluation/womd_evaluation.py": "mtr/datasets/waymo/waymo_eval.py",
    "data/dataset.py": (
        "mtr/datasets/waymo/waymo_interactive_dataset.py"
    ),
    "data/types.py": "mtr/datasets/waymo/waymo_types.py",
    "models/modules/joint_hypothesis_pipeline.py": (
        "mtr/models/motion_decoder/joint_hypothesis_pipeline.py"
    ),
    "models/modules/candidate_refinement.py": (
        "mtr/models/motion_decoder/candidate_refinement.py"
    ),
    "models/modules/spatial_context.py": (
        "mtr/models/motion_decoder/spatial_context.py"
    ),
    "models/modules/joint_reasoning.py": (
        "mtr/models/motion_decoder/joint_reasoning.py"
    ),
    "models/modules/jfer_decoder.py": (
        "mtr/models/motion_decoder/jfer_decoder.py"
    ),
    "models/modules/horizon_scoring.py": (
        "mtr/models/motion_decoder/horizon_scoring.py"
    ),
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def install(mtr_root: Path, config_path: Path) -> Path:
    mtr_root = mtr_root.expanduser().resolve()
    if not (mtr_root / "mtr/models/motion_decoder/mtr_decoder.py").is_file():
        raise RuntimeError(f"MTR checkout not found: {mtr_root}")

    mismatches = []
    for name, expected in BASE_FILES.items():
        path = mtr_root / name
        actual = _digest(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        paths = ", ".join(mismatches)
        raise RuntimeError(
            "The MTR checkout is incompatible with this release. "
            f"Mismatched files: {paths}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = mtr_root / ".jfer_backup" / stamp
    for source_name, destination_name in FILES.items():
        source = ROOT / source_name
        destination = mtr_root / destination_name
        if not source.is_file():
            raise RuntimeError(f"Missing release file: {source}")
        if destination.is_file() and _digest(destination) != _digest(source):
            backup = backup_root / destination_name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    installed_config = mtr_root / "tools/cfgs/waymo/jfer_womd.yaml"
    installed_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, installed_config)
    return installed_config
