# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022

import os

import torch
import torch.nn as nn

from .context_encoder import build_context_encoder
from .motion_decoder.jfer_decoder import JFERMTRDecoder


class MotionTransformer(nn.Module):
    """MTR backbone with the frozen-base JFER training policy."""

    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.context_encoder = build_context_encoder(
            self.model_cfg.CONTEXT_ENCODER
        )
        self.motion_decoder = JFERMTRDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER,
        )
        self.configure_trainable_parameters()

    def configure_trainable_parameters(self):
        """Apply the trainable-parameter policy of the released JFER run."""
        flow_prefix = (
            "motion_decoder.integrated_joint_world."
            "candidate_residual_flow."
        )
        score_prefix = flow_prefix + "horizon_score_head."
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                name.startswith(flow_prefix)
                and not name.startswith(score_prefix)
            )

        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("JFER training policy selected no parameters")
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
        if stage != "candidate_bank_residual_flow_geometry":
            raise ValueError(
                "The JFER release supports only the locked JFER geometry "
                f"training stage, got: {stage}"
            )

        # The locked experiment trains only JFER residual flow. Keep the
        # candidate generator and score/selection path deterministic.
        for module in self.modules():
            module.training = False
        self.training = True
        self.motion_decoder.training = True
        self.motion_decoder.integrated_joint_world.train_candidate_residual_flow_modules(
            score_only=False
        )
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
    architecture = config.get("ARCHITECTURE", "MotionTransformer")
    if architecture != "MotionTransformer":
        raise ValueError(
            "JFER packages one JFER architecture, got: " + architecture
        )
    return MotionTransformer(config)
