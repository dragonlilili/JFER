#!/usr/bin/env python3
"""Run the staged JFER training protocol."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_release_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["RELEASE"]


def stage_output(run_name: str, stage_name: str) -> Path:
    return (
        ROOT
        / "output"
        / "jfer"
        / "config"
        / run_name
        / stage_name
        / "ckpt"
        / "best_model.pth"
    )


def run_stage(
    *,
    config_path: Path,
    release: dict,
    stage: dict,
    initialization: Path,
    data_root: Path,
    run_name: str,
    gpu_list: str,
) -> Path:
    stage_name = str(stage["NAME"])
    devices = [device for device in gpu_list.split(",") if device.strip()]
    if not devices:
        raise ValueError("At least one CUDA device is required")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        "--module",
        "utils.training_runner",
        "--launcher",
        "pytorch",
        "--cfg_file",
        str(config_path),
        "--batch_size",
        str(release.get("GLOBAL_BATCH_SIZE", 80)),
        "--workers",
        str(release.get("WORKERS_PER_GPU", 8)),
        "--epochs",
        str(stage["EPOCHS"]),
        "--extra_tag",
        f"{run_name}/{stage_name}",
        "--pretrained_model",
        str(initialization),
        "--fix_random_seed",
        "--random_seed",
        str(release.get("SEED", 666)),
        "--set",
        "DATA_CONFIG.DATA_ROOT",
        str(data_root),
        "DATA_CONFIG.SAMPLE_INTERVAL.train",
        str(stage["TRAIN_SAMPLE_INTERVAL"]),
        "MODEL.MOTION_DECODER.INTENTION_POINTS_FILE",
        str(ROOT / "resources/intention_anchors.pkl"),
        "MODEL.MOTION_DECODER.INTEGRATED_JOINT_WORLD.TRAIN_STAGE",
        str(stage["TRAIN_STAGE"]),
        "OPTIMIZATION.LR",
        str(stage["LEARNING_RATE"]),
        "OPTIMIZATION.COSINE_T0_EPOCHS",
        str(stage["EPOCHS"]),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_list
    subprocess.run(command, cwd=ROOT, env=environment, check=True)

    checkpoint = stage_output(run_name, stage_name)
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Stage {stage_name} completed without a best checkpoint: "
            f"{checkpoint}"
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JFER")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--run-name")
    parser.add_argument("--gpus")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    release = load_release_config(config_path)
    data_root = (
        args.data_root or resolve_path(str(release["DATA_ROOT"]))
    ).resolve()
    initialization = (
        args.pretrained_model
        or resolve_path(str(release["PRETRAINED_MODEL"]))
    ).resolve()
    if not data_root.exists():
        raise RuntimeError(f"Processed WOMD directory not found: {data_root}")
    if not initialization.is_file():
        raise RuntimeError(
            "Pretrained forecasting model not found: " + str(initialization)
        )

    stages = list(release["TRAINING_STAGES"])
    available = {str(stage["NAME"]): stage for stage in stages}
    if args.stage != "all":
        if args.stage not in available:
            raise ValueError(
                f"Unknown stage {args.stage!r}; choose from "
                + ", ".join(available)
            )
        stages = [available[args.stage]]

    run_name = args.run_name or str(release.get("RUN_NAME", "jfer_train"))
    gpu_list = args.gpus or str(release.get("CUDA_VISIBLE_DEVICES", "0"))
    current = initialization
    for stage in stages:
        current = run_stage(
            config_path=config_path,
            release=release,
            stage=stage,
            initialization=current,
            data_root=data_root,
            run_name=run_name,
            gpu_list=gpu_list,
        )
    print(f"Training complete. Final checkpoint: {current}")


if __name__ == "__main__":
    main()
