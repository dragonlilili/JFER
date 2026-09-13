"""Multi-horizon scoring targets and reliability heads."""

import torch
import torch.nn as nn


def build_soft_map_credit_targets(horizon_match, horizon_cost, pair_valid):
    """Build per-horizon supervision with Waymo Soft mAP credit semantics.

    Only one matching mode can receive true-positive credit for a scenario and
    horizon. Other matching modes are ignored, exactly as they are during Soft
    mAP accumulation, instead of being trained as duplicate positives.
    """
    if horizon_match.ndim != 3 or horizon_cost.shape != horizon_match.shape:
        raise ValueError(
            "Expected horizon_match and horizon_cost with shape [B, M, H]"
        )
    if pair_valid.shape != (
        horizon_match.shape[0],
        horizon_match.shape[2],
    ):
        raise ValueError("Expected pair_valid with shape [B, H]")

    horizon_match = horizon_match.bool()
    pair_valid = pair_valid.bool()
    has_match = horizon_match.any(dim=1) & pair_valid
    matched_cost = horizon_cost.masked_fill(~horizon_match, float("inf"))
    credited_index = matched_cost.argmin(dim=1)

    credited_target = torch.zeros_like(horizon_match)
    credited_target.scatter_(1, credited_index[:, None], True)
    credited_target &= has_match[:, None]

    # Unmatched modes are false positives. Once one matching mode receives TP
    # credit, additional matching modes are omitted from the PR accumulator.
    supervision_mask = (
        (~horizon_match | credited_target) & pair_valid[:, None]
    )
    ignored_match = horizon_match & ~credited_target & pair_valid[:, None]
    return {
        "credited_target": credited_target,
        "supervision_mask": supervision_mask,
        "ignored_match": ignored_match,
        "has_match": has_match,
        "credited_index": credited_index,
    }


class WorldHorizonReliabilityHead(nn.Module):
    """Calibrate joint-mode confidence from candidate world responses.

    The head is permutation-equivariant over the joint modes. Its final layer
    is zero initialized so attaching it to a trained decoder is an exact
    no-op until reliability supervision updates the head.
    """

    def __init__(self, cfg, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = int(cfg.get("NUM_HORIZONS", 3))
        self.max_logit_delta = float(cfg.get("MAX_LOGIT_DELTA", 1.0))
        self.use_horizon_tokens = bool(
            cfg.get("USE_HORIZON_TOKENS", False)
        )

        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [1.0, 1.0, 1.0]),
            dtype=torch.float32,
        )
        if horizon_weights.numel() != self.num_horizons:
            raise ValueError(
                "HORIZON_WEIGHTS must contain one value per horizon"
            )
        if not torch.isfinite(horizon_weights).all() or (
            horizon_weights < 0
        ).any():
            raise ValueError("HORIZON_WEIGHTS must be finite and non-negative")
        if float(horizon_weights.sum()) <= 0.0:
            raise ValueError("HORIZON_WEIGHTS must have a positive sum")
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum(),
        )

        self.response_norm = nn.LayerNorm(self.hidden_dim)
        self.base_score_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=self.hidden_dim
            * int(cfg.get("FFN_MULTIPLIER", 2)),
            dropout=float(cfg.get("DROPOUT", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mode_set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(cfg.get("NUM_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        if self.use_horizon_tokens:
            self.horizon_embedding = nn.Parameter(
                torch.zeros(self.num_horizons, self.hidden_dim)
            )
            nn.init.normal_(self.horizon_embedding, std=0.02)
            self.horizon_token_head = nn.Linear(self.hidden_dim, 1)
            self.horizon_head = None
            nn.init.zeros_(self.horizon_token_head.weight)
            nn.init.zeros_(self.horizon_token_head.bias)
        else:
            self.horizon_embedding = None
            self.horizon_token_head = None
            self.horizon_head = nn.Linear(
                self.hidden_dim, self.num_horizons
            )
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(float(cfg.get("GATE_BIAS_INIT", -1.5)))
        )

    def forward(
        self,
        world_response,
        base_logits,
        horizon_world_response=None,
    ):
        if world_response.ndim != 3 or base_logits.ndim != 2:
            raise ValueError(
                "Expected world_response [B, M, H] and base_logits [B, M]"
            )
        if world_response.shape[:2] != base_logits.shape:
            raise ValueError("World response and base logits disagree on B/M")

        centered_score = base_logits - base_logits.mean(
            dim=-1, keepdim=True
        )
        score_scale = centered_score.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        normalized_score = centered_score / score_scale

        score_token = self.base_score_proj(normalized_score.unsqueeze(-1))
        if self.use_horizon_tokens:
            if horizon_world_response is None:
                raise ValueError(
                    "USE_HORIZON_TOKENS requires horizon_world_response"
                )
            expected = (
                base_logits.shape[0],
                base_logits.shape[1],
                self.num_horizons,
                self.hidden_dim,
            )
            if horizon_world_response.shape != expected:
                raise ValueError(
                    "Expected horizon_world_response with shape "
                    f"{expected}, got {tuple(horizon_world_response.shape)}"
                )
            mode_token = self.response_norm(horizon_world_response)
            mode_token = mode_token + score_token[:, :, None]
            mode_token = mode_token + self.horizon_embedding[None, None]
            batch_size, num_modes = base_logits.shape
            mode_token = mode_token.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_horizons,
                num_modes,
                self.hidden_dim,
            )
            mode_token = self.mode_set_encoder(mode_token)
            horizon_logits = self.horizon_token_head(mode_token)
            horizon_logits = horizon_logits.reshape(
                batch_size, self.num_horizons, num_modes
            ).permute(0, 2, 1).contiguous()
        else:
            mode_token = self.response_norm(world_response) + score_token
            mode_token = self.mode_set_encoder(mode_token)
            horizon_logits = self.horizon_head(mode_token)

        aggregate_logit = (
            horizon_logits * self.horizon_weights.view(1, 1, -1)
        ).sum(dim=-1)
        aggregate_logit = aggregate_logit - aggregate_logit.mean(
            dim=-1, keepdim=True
        )
        bounded_residual = self.max_logit_delta * torch.tanh(
            aggregate_logit
        )
        gate = torch.sigmoid(self.gate_logit)
        fused_logits = base_logits + gate * bounded_residual

        return {
            "base_logits": base_logits,
            "fused_logits": fused_logits,
            "horizon_logits": horizon_logits,
            "horizon_probability": torch.sigmoid(horizon_logits),
            "logit_residual": gate * bounded_residual,
            "gate": gate,
        }
