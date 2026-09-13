#!/usr/bin/env python3
"""Evaluate a JFER checkpoint on the WOMD interaction split."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JFER")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--gpus")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle).get("RELEASE", {})
    data_root = (args.data_root or _resolve(ROOT, settings["DATA_ROOT"])).resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    gpu_list = args.gpus or str(settings.get("CUDA_VISIBLE_DEVICES", "0"))
    devices = [item for item in gpu_list.split(",") if item.strip()]
    if not data_root.exists():
        raise RuntimeError(f"Processed WOMD directory not found: {data_root}")
    if not checkpoint.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_list
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        "--module",
        "evaluation.runner",
        "--launcher",
        "pytorch",
        "--cfg_file",
        str(config_path),
        "--batch_size",
        str(settings.get("GLOBAL_BATCH_SIZE", 80)),
        "--workers",
        str(settings.get("WORKERS_PER_GPU", 8)),
        "--extra_tag",
        "jfer_release",
        "--eval_tag",
        "official_validation",
        "--ckpt",
        str(checkpoint),
        "--set",
        "DATA_CONFIG.DATA_ROOT",
        str(data_root),
        "MODEL.MOTION_DECODER.INTENTION_POINTS_FILE",
        str(ROOT / "resources/intention_anchors.pkl"),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
