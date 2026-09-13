"""Top-level JFER model and checkpoint loading utilities."""

import os

import torch
import torch.nn as nn

from .encoder import build_context_encoder
from .decoder.jfer_decoder import JFERDecoder


class JFER(nn.Module):
    """Joint Future Exploration and Reasoning model."""

    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.context_encoder = build_context_encoder(
            self.model_cfg.CONTEXT_ENCODER
        )
        self.motion_decoder = JFERDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER,
        )
        self.configure_trainable_parameters()

    def configure_trainable_parameters(self):
        """Apply the parameter policy for the configured training stage."""
        world_cfg = self.model_cfg.MOTION_DECODER.INTEGRATED_JOINT_WORLD
        stage = str(world_cfg.get("TRAIN_STAGE", "full_joint")).lower()
        root = "motion_decoder.integrated_joint_world."
        if stage == "full_joint":
            prefixes = ("",)
            excluded = ()
        elif stage == "candidate_bank_warmup":
            prefixes = (root + "candidate_bank.",)
            excluded = ()
        elif stage == "candidate_bank_ap_calibration":
            prefixes = (root + "metric_scorer.",)
            excluded = ()
        elif stage == "candidate_bank_expansion":
            prefixes = (root + "candidate_expansion.",)
            excluded = ()
        elif stage == "candidate_bank_expansion_calibration":
            expansion = root + "candidate_expansion."
            prefixes = tuple(
                expansion + name
                for name in (
                    "cross_branch_score_head.",
                    "cross_branch_score_bias",
                    "cross_branch_score_gate_logit",
                )
            )
            excluded = ()
        elif stage == "candidate_bank_residual_flow_geometry":
            flow = root + "candidate_residual_flow."
            prefixes = (flow,)
            excluded = (flow + "horizon_score_head.",)
        else:
            raise ValueError(f"Unsupported JFER training stage: {stage}")

        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(
                name.startswith(prefix) for prefix in prefixes
            ) and not any(name.startswith(prefix) for prefix in excluded)

        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError(f"JFER stage {stage} selected no parameters")
        return trainable

    def forward(self, batch_dict):
        batch_dict = self.context_encoder(batch_dict)
        batch_dict = self.motion_decoder(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_loss()
            tb_dict.update({"loss": loss.item()})
            disp_dict.update({"loss": loss.item()})
            return loss, tb_dict, disp_dict
        return batch_dict

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            return self

        world_cfg = self.model_cfg.MOTION_DECODER.get(
            "INTEGRATED_JOINT_WORLD", None
        )
        if world_cfg is None or not world_cfg.get("ENABLED", False):
            return self

        stage = str(world_cfg.get("TRAIN_STAGE", "")).lower()
        if stage == "full_joint":
            return self

        for module in self.modules():
            module.training = False
        self.training = True
        self.motion_decoder.training = True
        joint_model = self.motion_decoder.integrated_joint_world
        if stage == "candidate_bank_warmup":
            joint_model.candidate_bank.train(True)
        elif stage == "candidate_bank_ap_calibration":
            joint_model.train_metric_scorer_modules()
        elif stage == "candidate_bank_expansion":
            joint_model.train_candidate_expansion_modules()
        elif stage == "candidate_bank_expansion_calibration":
            joint_model.train_candidate_expansion_calibration_modules()
        elif stage == "candidate_bank_residual_flow_geometry":
            joint_model.train_candidate_residual_flow_modules(score_only=False)
        else:
            raise ValueError(f"Unsupported JFER training stage: {stage}")
        return self

    def get_loss(self):
        return self.motion_decoder.get_loss()

    def load_params_with_optimizer(
        self, filename, to_cpu=False, optimizer=None, logger=None
    ):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        logger.info(
            "==> Loading parameters from checkpoint %s to %s",
            filename,
            "CPU" if to_cpu else "GPU",
        )
        checkpoint = torch.load(
            filename,
            map_location=torch.device("cpu") if to_cpu else None,
        )
        self.load_state_dict(checkpoint["model_state"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(
            "==> Done (loaded %d/%d)",
            len(checkpoint["model_state"]),
            len(checkpoint["model_state"]),
        )
        return checkpoint.get("it", 0.0), checkpoint.get("epoch", -1)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        logger.info(
            "==> Loading parameters from checkpoint %s to %s",
            filename,
            "CPU" if to_cpu else "GPU",
        )
        checkpoint = torch.load(
            filename,
            map_location=torch.device("cpu") if to_cpu else None,
        )
        target_state = self.state_dict()
        source_state = checkpoint["model_state"]
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        discarded = [key for key in source_state if key not in compatible]
        shape_mismatch = [
            key
            for key in discarded
            if key in target_state
            and target_state[key].shape != source_state[key].shape
        ]
        logger.info("Missing keys: %s", missing)
        logger.info("Unexpected keys: %s", unexpected)
        logger.info(
            "Discarded checkpoint keys: %d (shape mismatch: %d)",
            len(discarded),
            len(shape_mismatch),
        )
        logger.info(
            "==> Done (loaded %d/%d)", len(compatible), len(target_state)
        )
        return checkpoint.get("it", 0.0), checkpoint.get("epoch", -1)


def build_model(config):
    architecture = config.get("ARCHITECTURE", "JFER")
    if architecture != "JFER":
        raise ValueError(
            "JFER packages one JFER architecture, got: " + architecture
        )
    return JFER(config)
