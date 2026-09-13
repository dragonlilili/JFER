"""Focused training entry point for the released JFER model."""

import argparse
import datetime
import glob
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

from config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from data import build_dataloader
from models import model as model_utils
from utils import common as common_utils
from utils.training_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description="Train JFER")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extra_tag", default="default")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--pretrained_model", default=None)
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument("--tcp_port", type=int, default=18888)
    parser.add_argument("--without_sync_bn", action="store_true")
    parser.add_argument("--fix_random_seed", action="store_true")
    parser.add_argument("--random_seed", type=int, default=666)
    parser.add_argument("--ckpt_save_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument(
        "--local_rank", "--local-rank", dest="local_rank", type=int
    )
    parser.add_argument("--max_ckpt_save_num", type=int, default=6)
    parser.add_argument("--merge_all_iters_to_one_epoch", action="store_true")
    parser.add_argument("--not_eval_with_train", action="store_true")
    parser.add_argument("--logger_iter_interval", type=int, default=50)
    parser.add_argument("--ckpt_save_time_interval", type=int, default=300)
    parser.add_argument("--add_worker_init_fn", action="store_true")
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "jfer"
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def configure_math(opt_cfg):
    precision = str(opt_cfg.get("MATMUL_PRECISION", "high"))
    torch.set_float32_matmul_precision(precision)
    if torch.cuda.is_available():
        allow_tf32 = bool(opt_cfg.get("ALLOW_TF32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def build_optimizer(model, opt_cfg):
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise RuntimeError("No trainable JFER parameters")
    if opt_cfg.OPTIMIZER != "AdamW":
        raise ValueError("The released JFER protocol requires AdamW")
    return torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=float(opt_cfg.LR),
        weight_decay=float(opt_cfg.get("WEIGHT_DECAY", 0.0)),
    )


def build_scheduler(optimizer, dataloader, opt_cfg, last_epoch):
    scheduler_name = opt_cfg.get("SCHEDULER", None)
    if scheduler_name != "cosine_warmup":
        raise ValueError(
            "The released JFER protocol requires SCHEDULER=cosine_warmup"
        )

    warmup_iters = max(
        int(round(float(opt_cfg.get("WARMUP_EPOCHS", 0.5)) * len(dataloader))),
        1,
    )
    warmup_start = float(opt_cfg.get("WARMUP_START_RATIO", 0.1))
    cycle_iters = max(
        int(
            round(
                float(opt_cfg.get("COSINE_T0_EPOCHS", 6)) * len(dataloader)
            )
        ),
        1,
    )
    min_ratio = float(opt_cfg.get("MIN_LR_RATIO", 0.05))

    def lr_ratio(cur_iter):
        cur_iter = max(float(cur_iter), 0.0)
        if cur_iter < warmup_iters:
            progress = cur_iter / float(warmup_iters)
            return warmup_start + (1.0 - warmup_start) * progress
        progress = ((cur_iter - warmup_iters) % cycle_iters) / float(
            cycle_iters
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_sched.LambdaLR(
        optimizer, lr_lambda=lr_ratio, last_epoch=last_epoch
    )


def _latest_checkpoint(ckpt_dir):
    candidates = sorted(
        glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pth")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def main():
    args, cfg = parse_config()
    configure_math(cfg.OPTIMIZATION)

    if args.launcher == "none":
        dist_train = False
        total_gpus = 1
        args.without_sync_bn = True
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        total_gpus, cfg.LOCAL_RANK = getattr(
            common_utils, "init_dist_%s" % args.launcher
        )(args.tcp_port, args.local_rank, backend="nccl")
        dist_train = True

    if args.batch_size is None:
        per_gpu_batch = int(cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU)
    else:
        if args.batch_size % total_gpus != 0:
            raise ValueError("Global batch size must be divisible by GPU count")
        per_gpu_batch = args.batch_size // total_gpus
    args.epochs = (
        int(cfg.OPTIMIZATION.NUM_EPOCHS)
        if args.epochs is None
        else args.epochs
    )
    if args.fix_random_seed:
        common_utils.set_random_seed(args.random_seed)

    output_dir = (
        cfg.ROOT_DIR
        / "output"
        / cfg.EXP_GROUP_PATH
        / cfg.TAG
        / args.extra_tag
    )
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / (
        "log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info("********************** Start JFER training **********************")
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"))
    logger.info("global_batch=%d per_gpu_batch=%d", total_gpus * per_gpu_batch, per_gpu_batch)
    logger.info("seed=%d FP32=True TF32=%s", args.random_seed, torch.backends.cuda.matmul.allow_tf32)
    for key, value in vars(args).items():
        logger.info("%-24s %s", key, value)
    log_config_to_file(cfg, logger=logger)

    tb_log = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if cfg.LOCAL_RANK == 0
        else None
    )
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=per_gpu_batch,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
    )

    model = model_utils.build_model(config=cfg.MODEL)
    trainable_names = model.configure_trainable_parameters()
    if not args.without_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    logger.info("trainable_tensors=%d", len(trainable_names))
    logger.info(
        "trainable_parameters=%d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = 0
    accumulated_iter = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(
            filename=args.pretrained_model,
            to_cpu=dist_train,
            logger=logger,
        )
    if args.ckpt is not None:
        accumulated_iter, start_epoch = model.load_params_with_optimizer(
            args.ckpt,
            to_cpu=dist_train,
            optimizer=optimizer,
            logger=logger,
        )
        last_epoch = start_epoch + 1
    elif args.pretrained_model is None:
        latest = _latest_checkpoint(ckpt_dir)
        if latest is not None:
            accumulated_iter, start_epoch = model.load_params_with_optimizer(
                latest,
                to_cpu=dist_train,
                optimizer=optimizer,
                logger=logger,
            )
            last_epoch = start_epoch + 1

    scheduler = build_scheduler(
        optimizer,
        train_loader,
        cfg.OPTIMIZATION,
        last_epoch=last_epoch,
    )
    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
            find_unused_parameters=True,
        )

    test_loader = None
    if not args.not_eval_with_train:
        _, test_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            batch_size=per_gpu_batch,
            dist=dist_train,
            workers=args.workers,
            logger=logger,
            training=False,
        )
    eval_output_dir = output_dir / "eval" / "eval_with_train"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    train_model(
        model,
        optimizer,
        train_loader,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=accumulated_iter,
        rank=cfg.LOCAL_RANK,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        tb_log=tb_log,
        scheduler=scheduler,
        logger=logger,
        eval_output_dir=eval_output_dir,
        test_loader=test_loader,
        cfg=cfg,
        dist_train=dist_train,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        eval_interval=args.eval_interval,
    )
    logger.info("********************** End JFER training **********************")


if __name__ == "__main__":
    main()
