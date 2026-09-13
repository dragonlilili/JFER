"""Distributed evaluation entry point for a single JFER checkpoint."""

import argparse
import datetime
import os
import re
from pathlib import Path

import numpy as np
import torch

from config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from data import build_dataloader
from evaluation import evaluation_utils as eval_utils
from models import model as model_utils
from utils import common as common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Evaluate JFER")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra_tag", default="default")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument("--tcp_port", type=int, default=18888)
    parser.add_argument(
        "--local_rank", "--local-rank", dest="local_rank", type=int
    )
    parser.add_argument("--fix_random_seed", action="store_true")
    parser.add_argument("--eval_tag", default="default")
    parser.add_argument("--save_to_file", action="store_true")
    parser.add_argument(
        "--prediction_only",
        action="store_true",
        help="Save predictions without running ground-truth metrics.",
    )
    parser.add_argument(
        "--set", dest="set_cfgs", nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "jfer"
    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def evaluate_checkpoint(
    model, test_loader, args, output_dir, logger, epoch_id, dist_test=False
):
    iteration, epoch = model.load_params_from_file(
        filename=args.ckpt, logger=logger, to_cpu=dist_test
    )
    model.cuda()
    logger.info(
        "Loaded checkpoint for evaluation (epoch=%s, iter=%s)",
        epoch,
        iteration,
    )
    eval_utils.eval_one_epoch(
        cfg,
        model,
        test_loader,
        epoch_id,
        logger,
        dist_test=dist_test,
        result_dir=output_dir,
        save_to_file=args.save_to_file,
        prediction_only=args.prediction_only,
    )


def main():
    args, config = parse_config()
    if args.launcher == "none":
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        total_gpus, config.LOCAL_RANK = getattr(
            common_utils, "init_dist_%s" % args.launcher
        )(args.tcp_port, args.local_rank, backend="nccl")
        dist_test = True

    if args.batch_size is None:
        args.batch_size = config.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        if args.batch_size % total_gpus != 0:
            raise ValueError("Global batch size must be divisible by GPU count")
        args.batch_size //= total_gpus

    numbers = re.findall(r"\d+", args.ckpt)
    epoch_id = numbers[-1] if numbers else "checkpoint"
    output_dir = (
        config.ROOT_DIR
        / "output"
        / config.EXP_GROUP_PATH
        / config.TAG
        / args.extra_tag
        / "eval"
        / ("epoch_%s" % epoch_id)
        / args.eval_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / (
        "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=config.LOCAL_RANK)
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    )
    if dist_test:
        logger.info("total_batch_size=%d", total_gpus * args.batch_size)
    log_config_to_file(config, logger=logger)
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    _, test_loader, _ = build_dataloader(
        dataset_cfg=config.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    model = model_utils.build_model(config=config.MODEL)
    with torch.no_grad():
        evaluate_checkpoint(
            model,
            test_loader,
            args,
            output_dir,
            logger,
            epoch_id,
            dist_test=dist_test,
        )


if __name__ == "__main__":
    main()
