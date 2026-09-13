import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mtr.utils import common_utils
from .joint_reasoning import (
    IntegratedJointWorldDecoder,
    _build_mlp,
)
from .spatial_context import (
    SPATIAL_EVIDENCE_DIM,
    build_candidate_spatial_evidence,
)
from .horizon_scoring import build_soft_map_credit_targets
from .candidate_refinement import WorldConditionedCandidateResidualFlow


def official_pair_type_ids(pair_types, batch_size, device):
    if pair_types is None:
        return torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
    type_ids = []
    for pair in pair_types:
        names = {str(name) for name in pair}
        if "TYPE_CYCLIST" in names:
            type_ids.append(2)
        elif "TYPE_PEDESTRIAN" in names:
            type_ids.append(1)
        else:
            type_ids.append(0)
    return torch.tensor(type_ids, device=device, dtype=torch.long)


@torch.no_grad()
def official_ap_group_ids(input_dict, batch_size, device):
    """Build the type x GT-motion buckets used by interaction mAP."""
    type_ids = official_pair_type_ids(
        input_dict.get("pair_object_types"),
        batch_size,
        device,
    )
    gt_source = input_dict.get("pair_gt_trajs_src")
    if gt_source is None or gt_source.shape[0] != batch_size:
        return type_ids
    tracks = gt_source.to(device=device)
    if tracks.ndim != 4 or tracks.shape[1] != 2 or tracks.shape[-1] < 10:
        return type_ids

    current_step = min(10, tracks.shape[2] - 1)
    valid = tracks[..., 9].bool()
    future_steps = torch.arange(
        current_step + 1,
        tracks.shape[2],
        device=device,
        dtype=torch.long,
    )
    if future_steps.numel() == 0:
        return type_ids
    future_valid = valid.index_select(2, future_steps)
    last_step = future_steps[None, None].expand(
        batch_size, tracks.shape[1], -1
    ).masked_fill(~future_valid, -1).max(dim=-1).values
    track_valid = valid[:, :, current_step] & (last_step >= 0)
    gather_step = last_step.clamp_min(0)[..., None, None].expand(
        -1, -1, 1, tracks.shape[-1]
    )
    start = tracks[:, :, current_step]
    end = tracks.gather(2, gather_step).squeeze(2)

    delta = end[..., :2] - start[..., :2]
    displacement = torch.linalg.vector_norm(delta, dim=-1)
    start_heading = start[..., 6]
    heading_delta = end[..., 6] - start_heading
    heading_delta = torch.atan2(
        heading_delta.sin(), heading_delta.cos()
    )
    cos_heading = start_heading.cos()
    sin_heading = start_heading.sin()
    local_x = cos_heading * delta[..., 0] + sin_heading * delta[..., 1]
    local_y = -sin_heading * delta[..., 0] + cos_heading * delta[..., 1]
    start_speed = torch.linalg.vector_norm(start[..., 7:9], dim=-1)
    end_speed = torch.linalg.vector_norm(end[..., 7:9], dim=-1)

    behavior = torch.full_like(last_step, -1)
    stationary = (
        torch.maximum(start_speed, end_speed) < 2.0
    ) & (displacement < 3.0)
    straight_heading = heading_delta.abs() < (math.pi / 6.0)
    behavior = torch.where(stationary, 0, behavior)
    moving_straight = ~stationary & straight_heading
    behavior = torch.where(
        moving_straight & (local_y.abs() < 2.5), 1, behavior
    )
    behavior = torch.where(
        moving_straight & (local_y <= -2.5), 2, behavior
    )
    behavior = torch.where(
        moving_straight & (local_y >= 2.5), 3, behavior
    )
    turning = ~stationary & ~straight_heading
    behavior = torch.where(
        turning & (local_y < 0.0) & (local_x >= 0.0), 4, behavior
    )
    behavior = torch.where(
        turning & (local_y < 0.0) & (local_x < 0.0), 7, behavior
    )
    behavior = torch.where(
        turning & (local_y >= 0.0) & (local_x >= 0.0), 5, behavior
    )
    behavior = torch.where(
        turning & (local_y >= 0.0) & (local_x < 0.0), 6, behavior
    )
    behavior = behavior.masked_fill(~track_valid, -1)
    pair_behavior = behavior.max(dim=1).values
    pair_behavior = torch.where(
        pair_behavior == 7,
        torch.full_like(pair_behavior, 4),
        pair_behavior,
    )
    return torch.where(
        pair_behavior >= 0,
        type_ids * 7 + pair_behavior,
        torch.full_like(type_ids, -1),
    )


class WorldConditionedMetricScorer(nn.Module):
    """Calibrate final joint confidence without changing candidate geometry.

    The scorer reasons over all six candidates at each official Waymo horizon.
    Its deployment posterior is initialized as an exact copy of the verified
    base distribution, so attaching the module cannot alter trajectories or
    confidence before it receives metric-aligned supervision.
    """

    def __init__(
        self,
        cfg,
        hidden_dim,
        num_heads,
        measurement_steps,
        dt,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = len(measurement_steps)
        self.dt = float(dt)
        self.learn_temperature = bool(
            cfg.get("LEARN_TEMPERATURE", True)
        )
        self.learn_fusion_gate = bool(
            cfg.get("LEARN_FUSION_GATE", True)
        )
        self.use_type_expert_heads = bool(
            cfg.get("TYPE_EXPERT_HEADS", False)
        )
        self.use_temporal_motion_encoder = bool(
            cfg.get("TEMPORAL_MOTION_ENCODER", False)
        )
        self.use_type_specific_calibration = bool(
            cfg.get("TYPE_SPECIFIC_CALIBRATION", False)
        )
        self.use_trajectory_features = bool(
            cfg.get("USE_TRAJECTORY_FEATURES", True)
        )
        self.use_spatial_evidence = bool(
            cfg.get("USE_SPATIAL_EVIDENCE", False)
        )
        self.spatial_evidence_dim = int(
            cfg.get("SPATIAL_EVIDENCE_DIM", SPATIAL_EVIDENCE_DIM)
        )
        if (
            self.use_spatial_evidence
            and self.spatial_evidence_dim != SPATIAL_EVIDENCE_DIM
        ):
            raise ValueError(
                "METRIC_ALIGNED_SCORER.SPATIAL_EVIDENCE_DIM must be "
                f"{SPATIAL_EVIDENCE_DIM}"
            )
        spatial_groups = tuple(
            str(group).lower()
            for group in cfg.get(
                "SPATIAL_FEATURE_GROUPS",
                ("map", "neighbor", "pair", "motion", "traffic"),
            )
        )
        spatial_slices = {
            "map": (0, 8),
            "neighbor": (8, 14),
            "pair": (14, 18),
            "motion": (18, 19),
            "traffic": (19, 22),
        }
        unknown_groups = set(spatial_groups) - set(spatial_slices)
        if unknown_groups:
            raise ValueError(
                "Unknown spatial feature groups: "
                f"{sorted(unknown_groups)}"
            )
        spatial_feature_mask = torch.zeros(self.spatial_evidence_dim)
        for group in spatial_groups:
            start, end = spatial_slices[group]
            spatial_feature_mask[start:end] = 1.0
        self.register_buffer(
            "spatial_feature_mask",
            spatial_feature_mask,
            persistent=True,
        )
        self.max_log_temperature = math.log(
            max(float(cfg.get("MAX_TEMPERATURE", 2.0)), 1.0)
        )
        self.max_evidence_delta = float(
            cfg.get("MAX_EVIDENCE_DELTA", 2.0)
        )
        self.max_log_base_scale = math.log(
            max(float(cfg.get("MAX_BASE_LOGIT_SCALE", 2.0)), 1.0)
        )
        self.max_log_coverage_temperature = math.log(
            max(
                float(cfg.get("MAX_COVERAGE_TEMPERATURE", 2.0)),
                1.0,
            )
        )
        self.eps = 1e-6
        self.register_buffer(
            "measurement_steps",
            torch.as_tensor(measurement_steps, dtype=torch.long),
            persistent=False,
        )
        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [1.0] * self.num_horizons),
            dtype=torch.float32,
        )
        if horizon_weights.numel() != self.num_horizons:
            raise ValueError(
                "METRIC_ALIGNED_SCORER.HORIZON_WEIGHTS must match "
                "MEASUREMENT_STEPS"
            )
        if (
            not torch.isfinite(horizon_weights).all()
            or (horizon_weights < 0).any()
            or float(horizon_weights.sum()) <= 0.0
        ):
            raise ValueError(
                "Metric scorer horizon weights must be finite and positive"
            )
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum(),
            persistent=False,
        )

        self.trajectory_proj = _build_mlp(
            16, self.hidden_dim, self.hidden_dim, float(cfg.get("DROPOUT", 0.1))
        )
        if self.use_temporal_motion_encoder:
            temporal_dim = int(cfg.get("TEMPORAL_DIM", 64))
            self.temporal_stride = max(
                int(cfg.get("TEMPORAL_STRIDE", 4)), 1
            )
            self.temporal_input_proj = nn.Sequential(
                nn.Linear(16, temporal_dim),
                nn.LayerNorm(temporal_dim),
                nn.GELU(),
            )
            self.temporal_encoder = nn.GRU(
                input_size=temporal_dim,
                hidden_size=temporal_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_output_proj = nn.Sequential(
                nn.LayerNorm(temporal_dim * 2),
                nn.Linear(temporal_dim * 2, self.hidden_dim),
            )
        self.score_prior_proj = _build_mlp(
            3, self.hidden_dim, self.hidden_dim, float(cfg.get("DROPOUT", 0.1))
        )
        self.spatial_evidence_proj = (
            nn.Sequential(
                nn.LayerNorm(self.spatial_evidence_dim),
                _build_mlp(
                    self.spatial_evidence_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    float(cfg.get("DROPOUT", 0.1)),
                ),
            )
            if self.use_spatial_evidence
            else None
        )
        if self.spatial_evidence_proj is not None:
            nn.init.zeros_(self.spatial_evidence_proj[1][-1].weight)
            nn.init.zeros_(self.spatial_evidence_proj[1][-1].bias)
        self.use_direct_spatial_energy = bool(
            self.use_spatial_evidence
            and cfg.get("DIRECT_SPATIAL_ENERGY", True)
        )
        self.max_spatial_energy_delta = float(
            cfg.get("MAX_SPATIAL_ENERGY_DELTA", 2.0)
        )
        if self.use_direct_spatial_energy:
            spatial_type_dim = int(cfg.get("SPATIAL_TYPE_DIM", 8))
            self.spatial_energy_norm = nn.LayerNorm(
                self.spatial_evidence_dim
            )
            self.spatial_energy_type_embedding = nn.Embedding(
                3, spatial_type_dim
            )
            self.spatial_energy_head = _build_mlp(
                self.spatial_evidence_dim + spatial_type_dim,
                max(self.hidden_dim // 2, 32),
                1,
                float(cfg.get("DROPOUT", 0.1)),
            )
            nn.init.zeros_(self.spatial_energy_head[-1].weight)
            nn.init.zeros_(self.spatial_energy_head[-1].bias)
        else:
            self.spatial_energy_norm = None
            self.spatial_energy_type_embedding = None
            self.spatial_energy_head = None
        self.use_relative_spatial_energy = bool(
            self.use_spatial_evidence
            and cfg.get("RELATIVE_SPATIAL_ENERGY", False)
        )
        self.max_relative_spatial_delta = float(
            cfg.get("MAX_RELATIVE_SPATIAL_DELTA", 1.0)
        )
        if self.use_relative_spatial_energy:
            relative_spatial_dim = self.spatial_evidence_dim * 3
            relative_hidden_dim = int(
                cfg.get(
                    "RELATIVE_SPATIAL_HIDDEN_DIM",
                    max(self.hidden_dim // 2, 64),
                )
            )
            self.relative_spatial_norm = nn.LayerNorm(
                relative_spatial_dim
            )
            self.relative_spatial_heads = nn.ModuleList(
                [
                    _build_mlp(
                        relative_spatial_dim,
                        relative_hidden_dim,
                        1,
                        float(cfg.get("DROPOUT", 0.1)),
                    )
                    for _ in range(3)
                ]
            )
            for head in self.relative_spatial_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.relative_spatial_norm = None
            self.relative_spatial_heads = None
        self.response_norm = nn.LayerNorm(self.hidden_dim)
        self.joint_context_norm = nn.LayerNorm(self.hidden_dim)
        self.scene_context_norm = nn.LayerNorm(self.hidden_dim)
        self.pair_type_embedding = nn.Embedding(3, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.num_horizons, self.hidden_dim)
        )
        nn.init.zeros_(self.pair_type_embedding.weight)
        nn.init.normal_(self.horizon_embedding, std=0.02)

        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=self.hidden_dim
            * int(cfg.get("FFN_MULTIPLIER", 2)),
            dropout=float(cfg.get("DROPOUT", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(cfg.get("NUM_SET_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        horizon_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=self.hidden_dim
            * int(cfg.get("FFN_MULTIPLIER", 2)),
            dropout=float(cfg.get("DROPOUT", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.horizon_encoder = nn.TransformerEncoder(
            horizon_layer,
            num_layers=int(cfg.get("NUM_HORIZON_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.use_comparative_final_ranker = bool(
            cfg.get("COMPARATIVE_FINAL_RANKER", False)
        )
        self.max_comparative_logit_delta = float(
            cfg.get("MAX_COMPARATIVE_LOGIT_DELTA", 1.0)
        )
        if self.use_comparative_final_ranker:
            comparative_dim = int(
                cfg.get("COMPARATIVE_HIDDEN_DIM", 128)
            )
            comparative_input_dim = (
                self.num_horizons * self.hidden_dim + 3
            )
            if self.use_spatial_evidence:
                comparative_input_dim += (
                    self.num_horizons
                    * self.spatial_evidence_dim
                    * 3
                )
            self.comparative_input_norm = nn.LayerNorm(
                comparative_input_dim
            )
            self.comparative_input_proj = _build_mlp(
                comparative_input_dim,
                comparative_dim,
                comparative_dim,
                float(cfg.get("DROPOUT", 0.1)),
            )
            comparative_layer = nn.TransformerEncoderLayer(
                d_model=comparative_dim,
                nhead=int(
                    cfg.get("COMPARATIVE_NUM_HEADS", 4)
                ),
                dim_feedforward=comparative_dim
                * int(cfg.get("COMPARATIVE_FFN_MULTIPLIER", 2)),
                dropout=float(cfg.get("DROPOUT", 0.1)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.comparative_set_encoder = nn.TransformerEncoder(
                comparative_layer,
                num_layers=int(
                    cfg.get("COMPARATIVE_NUM_LAYERS", 1)
                ),
                norm=nn.LayerNorm(comparative_dim),
            )
            self.comparative_type_heads = nn.ModuleList(
                [
                    _build_mlp(
                        comparative_dim,
                        max(comparative_dim // 2, 32),
                        1,
                        float(cfg.get("DROPOUT", 0.1)),
                    )
                    for _ in range(3)
                ]
            )
            for head in self.comparative_type_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.comparative_input_norm = None
            self.comparative_input_proj = None
            self.comparative_set_encoder = None
            self.comparative_type_heads = None
        self.use_score_multiset_permutation = bool(
            cfg.get("SCORE_MULTISET_PERMUTATION", False)
        )
        self.max_permutation_utility_delta = float(
            cfg.get("MAX_PERMUTATION_UTILITY_DELTA", 4.0)
        )
        if self.use_score_multiset_permutation:
            permutation_dim = int(
                cfg.get("PERMUTATION_HIDDEN_DIM", 192)
            )
            permutation_input_dim = (
                self.num_horizons * self.hidden_dim + 3
            )
            if self.use_spatial_evidence:
                permutation_input_dim += (
                    self.num_horizons
                    * self.spatial_evidence_dim
                    * 3
                )
            self.permutation_input_norm = nn.LayerNorm(
                permutation_input_dim
            )
            self.permutation_input_proj = _build_mlp(
                permutation_input_dim,
                permutation_dim,
                permutation_dim,
                float(cfg.get("DROPOUT", 0.1)),
            )
            permutation_layer = nn.TransformerEncoderLayer(
                d_model=permutation_dim,
                nhead=int(cfg.get("PERMUTATION_NUM_HEADS", 4)),
                dim_feedforward=permutation_dim
                * int(cfg.get("PERMUTATION_FFN_MULTIPLIER", 3)),
                dropout=float(cfg.get("DROPOUT", 0.1)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.permutation_set_encoder = nn.TransformerEncoder(
                permutation_layer,
                num_layers=int(
                    cfg.get("PERMUTATION_NUM_LAYERS", 2)
                ),
                norm=nn.LayerNorm(permutation_dim),
            )
            self.permutation_type_heads = nn.ModuleList(
                [
                    _build_mlp(
                        permutation_dim,
                        max(permutation_dim // 2, 32),
                        1,
                        float(cfg.get("DROPOUT", 0.1)),
                    )
                    for _ in range(3)
                ]
            )
            # An attached permutation head is an exact no-op. The utility
            # initially has the same ordering as the verified confidence and
            # the original score multiset is assigned back unchanged.
            for head in self.permutation_type_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.permutation_input_norm = None
            self.permutation_input_proj = None
            self.permutation_set_encoder = None
            self.permutation_type_heads = None
        if self.use_type_expert_heads:
            self.match_head = nn.ModuleList(
                [
                    _build_mlp(
                        self.hidden_dim,
                        self.hidden_dim,
                        1,
                        float(cfg.get("DROPOUT", 0.1)),
                    )
                    for _ in range(3)
                ]
            )
            self.scene_match_head = nn.ModuleList(
                [
                    _build_mlp(
                        self.hidden_dim * 2,
                        self.hidden_dim,
                        1,
                        float(cfg.get("DROPOUT", 0.1)),
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.match_head = _build_mlp(
                self.hidden_dim,
                self.hidden_dim,
                1,
                float(cfg.get("DROPOUT", 0.1)),
            )
            self.scene_match_head = _build_mlp(
                self.hidden_dim * 2,
                self.hidden_dim,
                1,
                float(cfg.get("DROPOUT", 0.1)),
            )
        self.temperature_head = (
            _build_mlp(
                self.hidden_dim * 2,
                self.hidden_dim,
                1,
                float(cfg.get("DROPOUT", 0.1)),
            )
            if self.learn_temperature
            else None
        )
        self.fusion_gate_head = (
            _build_mlp(
                self.hidden_dim * 2,
                self.hidden_dim,
                1,
                float(cfg.get("DROPOUT", 0.1)),
            )
            if self.learn_fusion_gate
            else None
        )
        zero_initialized_heads = []
        if self.use_type_expert_heads:
            zero_initialized_heads.extend(self.match_head)
            zero_initialized_heads.extend(self.scene_match_head)
        else:
            zero_initialized_heads.extend(
                [self.match_head, self.scene_match_head]
            )
        if self.temperature_head is not None:
            zero_initialized_heads.append(self.temperature_head)
        for head in zero_initialized_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.fusion_gate_head is not None:
            nn.init.zeros_(self.fusion_gate_head[-1].weight)
            nn.init.constant_(
                self.fusion_gate_head[-1].bias,
                float(cfg.get("FUSION_GATE_BIAS_INIT", -1.5)),
            )
        evidence_scale = max(
            float(cfg.get("EVIDENCE_SCALE_INIT", 1.0)), 1e-4
        )
        self.evidence_scale_unconstrained = nn.Parameter(
            torch.tensor(math.log(math.expm1(evidence_scale)))
        )
        self.base_scale_unconstrained = nn.Parameter(torch.zeros(()))
        coverage_strength = min(
            max(
                float(
                    cfg.get(
                        "COVERAGE_TEMPERATURE_STRENGTH_INIT",
                        0.75,
                    )
                ),
                0.0,
            ),
            0.999,
        )
        self.coverage_temperature_strength_unconstrained = nn.Parameter(
            torch.tensor(math.atanh(coverage_strength))
        )
        if self.use_type_specific_calibration:
            initial_horizon_weights = self.horizon_weights.clamp_min(
                self.eps
            ).log()
            self.type_horizon_weight_logits = nn.Parameter(
                initial_horizon_weights[None].repeat(3, 1)
            )
            self.type_evidence_scale_delta = nn.Parameter(
                torch.zeros(3)
            )
            self.type_base_scale_delta = nn.Parameter(torch.zeros(3))
            self.type_coverage_strength_delta = nn.Parameter(
                torch.zeros(3)
            )

    @staticmethod
    def _normalize_score(score):
        centered = score - score.mean(dim=-1, keepdim=True)
        scale = centered.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        return centered / scale

    def _trajectory_features(self, trajectories):
        steps = self.measurement_steps.to(trajectories.device).clamp(
            0, trajectories.shape[3] - 1
        )
        previous_steps = (steps - 5).clamp_min(0)
        position = trajectories.index_select(3, steps).permute(
            0, 1, 3, 2, 4
        )
        previous = trajectories.index_select(3, previous_steps).permute(
            0, 1, 3, 2, 4
        )
        elapsed = (
            (steps - previous_steps).clamp_min(1).type_as(trajectories)
            * self.dt
        )
        velocity = (position - previous) / elapsed[None, None, :, None, None]
        relative_position = position[..., 1, :] - position[..., 0, :]
        relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(0.5)
        speed = torch.linalg.vector_norm(
            velocity, dim=-1
        )
        return torch.cat(
            [
                position.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                relative_position / 20.0,
                relative_velocity / 20.0,
                distance / 20.0,
                closing_speed / 10.0,
                speed / 20.0,
            ],
            dim=-1,
        )

    def _temporal_trajectory_features(self, trajectories):
        position = trajectories.permute(0, 1, 3, 2, 4)
        velocity = torch.diff(
            position, dim=2, prepend=position[:, :, :1]
        ) / self.dt
        relative_position = position[..., 1, :] - position[..., 0, :]
        relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(0.5)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        sequence = torch.cat(
            [
                position.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                relative_position / 20.0,
                relative_velocity / 20.0,
                distance / 20.0,
                closing_speed / 10.0,
                speed / 20.0,
            ],
            dim=-1,
        )
        sequence = sequence[:, :, :: self.temporal_stride]
        batch_size, num_modes, num_steps, _ = sequence.shape
        sequence = self.temporal_input_proj(sequence).reshape(
            batch_size * num_modes, num_steps, -1
        )
        sequence, _ = self.temporal_encoder(sequence)
        sampled_steps = (
            self.measurement_steps.to(trajectories.device)
            // self.temporal_stride
        ).clamp(0, num_steps - 1)
        sequence = sequence.index_select(1, sampled_steps).reshape(
            batch_size, num_modes, self.num_horizons, -1
        )
        return self.temporal_output_proj(sequence)

    def _relative_spatial_features(self, spatial_evidence):
        """Compare each candidate with the alternatives in the same scene."""
        mode_mean = spatial_evidence.mean(dim=1, keepdim=True)
        mode_scale = spatial_evidence.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(0.05)
        standardized = (spatial_evidence - mode_mean) / mode_scale

        pairwise_delta = (
            spatial_evidence[:, :, None]
            - spatial_evidence[:, None, :]
        ) / mode_scale[:, :, None]
        soft_rank = torch.tanh(pairwise_delta).mean(dim=2)
        stabilized_absolute = torch.sign(spatial_evidence) * torch.log1p(
            spatial_evidence.abs()
        )
        return torch.cat(
            [stabilized_absolute, standardized, soft_rank], dim=-1
        )

    def forward(
        self,
        world_response,
        trajectories,
        base_logits,
        marginal_log_prior,
        bank_log_prior,
        pair_type_ids,
        joint_context=None,
        scene_context=None,
        spatial_evidence=None,
    ):
        batch_size, num_modes = base_logits.shape
        if world_response.shape != (
            batch_size,
            num_modes,
            self.hidden_dim,
        ):
            raise ValueError("World response and base logits disagree")
        if marginal_log_prior.shape != base_logits.shape:
            raise ValueError("Marginal prior must match base logits")
        if bank_log_prior.shape != base_logits.shape:
            raise ValueError("Candidate-bank prior must match base logits")
        if joint_context is None:
            joint_context = world_response.new_zeros(
                batch_size, num_modes, self.hidden_dim
            )
        if joint_context.shape != world_response.shape:
            raise ValueError("Joint context and world response disagree")
        if scene_context is None:
            scene_context = world_response.new_zeros(
                batch_size, self.hidden_dim
            )
        if scene_context.shape != (
            batch_size,
            self.hidden_dim,
        ):
            raise ValueError("Scene context has an unexpected shape")

        score_features = torch.stack(
            [
                self._normalize_score(base_logits),
                self._normalize_score(marginal_log_prior),
                self._normalize_score(bank_log_prior),
            ],
            dim=-1,
        )
        pair_type_ids = pair_type_ids.to(
            base_logits.device
        ).long().clamp(0, 2)
        hidden = (
            self.response_norm(world_response)[:, :, None]
            + self.joint_context_norm(joint_context)[:, :, None]
            + self.scene_context_norm(scene_context)[:, None, None]
            + self.score_prior_proj(score_features)[:, :, None]
            + self.pair_type_embedding(
                pair_type_ids
            )[:, None, None]
            + self.horizon_embedding[None, None]
        )
        if self.use_trajectory_features:
            hidden = hidden + self.trajectory_proj(
                self._trajectory_features(trajectories)
            )
        if self.use_temporal_motion_encoder:
            hidden = hidden + self._temporal_trajectory_features(
                trajectories
            )
        masked_spatial_evidence = None
        relative_spatial_features = None
        if self.use_spatial_evidence:
            expected_shape = (
                batch_size,
                num_modes,
                self.num_horizons,
                self.spatial_evidence_dim,
            )
            if (
                spatial_evidence is None
                or spatial_evidence.shape != expected_shape
            ):
                actual_shape = (
                    None
                    if spatial_evidence is None
                    else tuple(spatial_evidence.shape)
                )
                raise ValueError(
                    "Spatial evidence has shape "
                    f"{actual_shape}, expected {expected_shape}"
                )
            masked_spatial_evidence = (
                spatial_evidence.type_as(hidden)
                * self.spatial_feature_mask.type_as(hidden)
            )
            hidden = hidden + self.spatial_evidence_proj(
                masked_spatial_evidence
            )
            if (
                self.use_relative_spatial_energy
                or self.use_comparative_final_ranker
            ):
                relative_spatial_features = (
                    self._relative_spatial_features(
                        masked_spatial_evidence
                    )
                )
        hidden_dim = hidden.shape[-1]
        candidate_tokens = hidden.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_horizons, num_modes, hidden_dim
        )
        candidate_tokens = self.candidate_set_encoder(candidate_tokens)
        candidate_tokens = candidate_tokens.reshape(
            batch_size, self.num_horizons, num_modes, hidden_dim
        ).permute(0, 2, 1, 3)
        horizon_tokens = self.horizon_encoder(
            candidate_tokens.reshape(
                batch_size * num_modes,
                self.num_horizons,
                hidden_dim,
            )
        ).reshape(
            batch_size, num_modes, self.num_horizons, hidden_dim
        )

        if self.use_type_expert_heads:
            expert_horizon_logits = torch.stack(
                [
                    head(horizon_tokens).squeeze(-1)
                    for head in self.match_head
                ],
                dim=1,
            )
            horizon_logits = expert_horizon_logits[
                torch.arange(batch_size, device=base_logits.device),
                pair_type_ids,
            ]
        else:
            horizon_logits = self.match_head(
                horizon_tokens
            ).squeeze(-1)
        spatial_energy_delta = horizon_logits.new_zeros(
            horizon_logits.shape
        )
        if self.use_direct_spatial_energy:
            spatial_type = self.spatial_energy_type_embedding(
                pair_type_ids
            )[:, None, None].expand(
                -1, num_modes, self.num_horizons, -1
            )
            raw_spatial_energy = self.spatial_energy_head(
                torch.cat(
                    [
                        self.spatial_energy_norm(
                            masked_spatial_evidence
                        ),
                        spatial_type,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            raw_spatial_energy = (
                raw_spatial_energy
                - raw_spatial_energy.mean(dim=1, keepdim=True)
            )
            spatial_energy_delta = (
                self.max_spatial_energy_delta
                * torch.tanh(
                    raw_spatial_energy
                    / max(self.max_spatial_energy_delta, self.eps)
                )
            )
            horizon_logits = horizon_logits + spatial_energy_delta
        relative_spatial_delta = horizon_logits.new_zeros(
            horizon_logits.shape
        )
        if self.use_relative_spatial_energy:
            relative_features = self.relative_spatial_norm(
                relative_spatial_features
            )
            expert_relative_energy = torch.stack(
                [
                    head(relative_features).squeeze(-1)
                    for head in self.relative_spatial_heads
                ],
                dim=1,
            )
            raw_relative_energy = expert_relative_energy[
                torch.arange(batch_size, device=base_logits.device),
                pair_type_ids,
            ]
            raw_relative_energy = (
                raw_relative_energy
                - raw_relative_energy.mean(dim=1, keepdim=True)
            )
            relative_spatial_delta = (
                self.max_relative_spatial_delta
                * torch.tanh(
                    raw_relative_energy
                    / max(self.max_relative_spatial_delta, self.eps)
                )
            )
            horizon_logits = horizon_logits + relative_spatial_delta
        scene_tokens = torch.cat(
            [
                horizon_tokens.mean(dim=1),
                horizon_tokens.max(dim=1).values,
            ],
            dim=-1,
        )
        if self.use_type_expert_heads:
            expert_scene_logits = torch.stack(
                [
                    head(scene_tokens).squeeze(-1)
                    for head in self.scene_match_head
                ],
                dim=1,
            )
            scene_match_logits = expert_scene_logits[
                torch.arange(batch_size, device=base_logits.device),
                pair_type_ids,
            ]
        else:
            scene_match_logits = self.scene_match_head(
                scene_tokens
            ).squeeze(-1)
        scene_match_probability = torch.sigmoid(scene_match_logits)
        if self.use_type_specific_calibration:
            horizon_weights = torch.softmax(
                self.type_horizon_weight_logits, dim=-1
            )[pair_type_ids]
        else:
            horizon_weights = self.horizon_weights.type_as(
                scene_tokens
            )[None].expand(batch_size, -1)
        scene_summary = (
            scene_tokens * horizon_weights[:, :, None]
        ).sum(dim=1)
        if self.temperature_head is None:
            score_temperature = base_logits.new_ones(batch_size)
        else:
            log_temperature = self.max_log_temperature * torch.tanh(
                self.temperature_head(scene_summary).squeeze(-1)
            )
            score_temperature = log_temperature.exp()

        evidence = (
            horizon_logits * horizon_weights[:, None]
        ).sum(dim=-1)
        evidence = evidence - evidence.mean(dim=-1, keepdim=True)
        evidence_scale_input = self.evidence_scale_unconstrained
        if self.use_type_specific_calibration:
            evidence_scale_input = (
                evidence_scale_input
                + self.type_evidence_scale_delta[pair_type_ids]
            )
        evidence_scale = F.softplus(evidence_scale_input)
        evidence_delta = self.max_evidence_delta * torch.tanh(
            evidence_scale[..., None]
            * evidence
            / self.max_evidence_delta
        )
        coverage_probability = (
            scene_match_probability * horizon_weights
        ).sum(dim=-1)
        coverage_strength_input = (
            self.coverage_temperature_strength_unconstrained
        )
        if self.use_type_specific_calibration:
            coverage_strength_input = (
                coverage_strength_input
                + self.type_coverage_strength_delta[pair_type_ids]
            )
        coverage_temperature_strength = torch.tanh(
            coverage_strength_input
        )
        coverage_log_temperature = (
            -self.max_log_coverage_temperature
            * coverage_temperature_strength
            * (2.0 * coverage_probability - 1.0)
        )
        coverage_temperature = coverage_log_temperature.exp()
        score_temperature = score_temperature * coverage_temperature
        base_scale_input = self.base_scale_unconstrained
        if self.use_type_specific_calibration:
            base_scale_input = (
                base_scale_input
                + self.type_base_scale_delta[pair_type_ids]
            )
        base_logit_scale = torch.exp(
            self.max_log_base_scale * torch.tanh(base_scale_input)
        )
        posterior_logits = (
            base_logit_scale[..., None]
            * base_logits
            / score_temperature[:, None]
            + evidence_delta
        )
        if self.fusion_gate_head is None:
            fusion_gate = base_logits.new_ones(batch_size)
        else:
            fusion_gate = torch.sigmoid(
                self.fusion_gate_head(scene_summary).squeeze(-1)
            )
        joint_logits = base_logits + fusion_gate[:, None] * (
            posterior_logits - base_logits
        )
        comparative_logit_delta = joint_logits.new_zeros(
            joint_logits.shape
        )
        if self.use_comparative_final_ranker:
            comparative_features = [
                horizon_tokens.flatten(start_dim=-2),
                score_features,
            ]
            if relative_spatial_features is not None:
                comparative_features.append(
                    relative_spatial_features.flatten(start_dim=-2)
                )
            comparative_tokens = self.comparative_input_proj(
                self.comparative_input_norm(
                    torch.cat(comparative_features, dim=-1)
                )
            )
            comparative_tokens = self.comparative_set_encoder(
                comparative_tokens
            )
            expert_comparative_logits = torch.stack(
                [
                    head(comparative_tokens).squeeze(-1)
                    for head in self.comparative_type_heads
                ],
                dim=1,
            )
            raw_comparative_logits = expert_comparative_logits[
                torch.arange(batch_size, device=base_logits.device),
                pair_type_ids,
            ]
            raw_comparative_logits = (
                raw_comparative_logits
                - raw_comparative_logits.mean(dim=-1, keepdim=True)
            )
            comparative_logit_delta = (
                self.max_comparative_logit_delta
                * torch.tanh(
                    raw_comparative_logits
                    / max(
                        self.max_comparative_logit_delta,
                        self.eps,
                    )
                )
            )
            joint_logits = joint_logits + comparative_logit_delta
        pre_permutation_logits = joint_logits
        permutation_utility = joint_logits
        permutation_utility_delta = joint_logits.new_zeros(
            joint_logits.shape
        )
        permutation_order = None
        if self.use_score_multiset_permutation:
            permutation_features = [
                horizon_tokens.flatten(start_dim=-2),
                score_features,
            ]
            if relative_spatial_features is not None:
                permutation_features.append(
                    relative_spatial_features.flatten(start_dim=-2)
                )
            permutation_tokens = self.permutation_input_proj(
                self.permutation_input_norm(
                    torch.cat(permutation_features, dim=-1)
                )
            )
            permutation_tokens = self.permutation_set_encoder(
                permutation_tokens
            )
            expert_permutation_delta = torch.stack(
                [
                    head(permutation_tokens).squeeze(-1)
                    for head in self.permutation_type_heads
                ],
                dim=1,
            )
            raw_permutation_delta = expert_permutation_delta[
                torch.arange(batch_size, device=base_logits.device),
                pair_type_ids,
            ]
            raw_permutation_delta = (
                raw_permutation_delta
                - raw_permutation_delta.mean(dim=-1, keepdim=True)
            )
            permutation_utility_delta = (
                self.max_permutation_utility_delta
                * torch.tanh(
                    raw_permutation_delta
                    / max(
                        self.max_permutation_utility_delta,
                        self.eps,
                    )
                )
            )
            permutation_utility = (
                pre_permutation_logits.detach()
                + permutation_utility_delta
            )
            permutation_order = permutation_utility.argsort(
                dim=-1, descending=True
            )
            sorted_logits = pre_permutation_logits.sort(
                dim=-1, descending=True
            ).values
            # Only the assignment changes. This is deliberately different
            # from a residual confidence head: calibration magnitude and the
            # per-scene entropy of the verified predictor remain invariant.
            joint_logits = torch.zeros_like(sorted_logits).scatter(
                1, permutation_order, sorted_logits
            )
        probability = torch.softmax(joint_logits, dim=-1)
        return {
            "joint_logits": joint_logits,
            "base_logits": base_logits,
            "horizon_logits": horizon_logits,
            "scene_match_logits": scene_match_logits,
            "scene_match_probability": scene_match_probability,
            "horizon_weights": horizon_weights,
            "coverage_probability": coverage_probability,
            "score_temperature": score_temperature,
            "coverage_temperature": coverage_temperature,
            "coverage_temperature_strength": (
                coverage_temperature_strength
            ),
            "base_logit_scale": base_logit_scale,
            "evidence_scale": evidence_scale,
            "evidence_delta": evidence_delta,
            "spatial_energy_delta": spatial_energy_delta,
            "relative_spatial_delta": relative_spatial_delta,
            "comparative_logit_delta": comparative_logit_delta,
            "pre_permutation_logits": pre_permutation_logits,
            "permutation_utility": permutation_utility,
            "permutation_utility_delta": permutation_utility_delta,
            "permutation_order": permutation_order,
            "fusion_gate": fusion_gate,
            "probability": probability,
        }


class DeployedSetUtilityScorer(nn.Module):
    """Rank the six trajectories that are actually submitted to Waymo.

    Candidate admission stays frozen and independent from this module.  The
    scorer is permutation equivariant over the deployed set and predicts a
    zero-initialized residual, preserving the verified checkpoint exactly at
    initialization while allowing every deployed mode to change rank.
    """

    def __init__(self, cfg, hidden_dim, num_heads, measurement_steps, dt):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = len(measurement_steps)
        self.dt = float(dt)
        self.max_logit_delta = float(cfg.get("MAX_LOGIT_DELTA", 2.0))
        self.eps = 1e-6
        dropout = float(cfg.get("DROPOUT", 0.1))
        self.register_buffer(
            "measurement_steps",
            torch.as_tensor(measurement_steps, dtype=torch.long),
            persistent=False,
        )
        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [1.0] * self.num_horizons),
            dtype=torch.float32,
        )
        if horizon_weights.numel() != self.num_horizons:
            raise ValueError(
                "DEPLOYED_SET_SCORER.HORIZON_WEIGHTS must match "
                "MEASUREMENT_STEPS"
            )
        if (
            not torch.isfinite(horizon_weights).all()
            or (horizon_weights < 0).any()
            or float(horizon_weights.sum()) <= 0.0
        ):
            raise ValueError("Invalid deployed-set horizon weights")
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum(),
            persistent=False,
        )

        self.world_norm = nn.LayerNorm(self.hidden_dim)
        self.scene_norm = nn.LayerNorm(self.hidden_dim)
        self.trajectory_proj = _build_mlp(
            16, self.hidden_dim, self.hidden_dim, dropout
        )
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(SPATIAL_EVIDENCE_DIM),
            _build_mlp(
                SPATIAL_EVIDENCE_DIM,
                self.hidden_dim,
                self.hidden_dim,
                dropout,
            ),
        )
        self.score_proj = _build_mlp(
            3, self.hidden_dim, self.hidden_dim, dropout
        )
        self.type_embedding = nn.Embedding(3, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.num_horizons, self.hidden_dim)
        )
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.normal_(self.horizon_embedding, std=0.02)

        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=(
                self.hidden_dim * int(cfg.get("FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(cfg.get("NUM_SET_LAYERS", 2)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        horizon_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=(
                self.hidden_dim * int(cfg.get("FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.horizon_encoder = nn.TransformerEncoder(
            horizon_layer,
            num_layers=int(cfg.get("NUM_HORIZON_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.utility_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            dropout,
        )
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)
        self.gate_logit = nn.Parameter(
            torch.tensor(float(cfg.get("GATE_BIAS_INIT", -1.0)))
        )

        # The deployed-set permutation is deliberately downstream of candidate
        # admission.  It can only associate the already verified six score
        # values with the already selected six trajectories.  This separates
        # confidence ranking from coverage and geometry by construction.
        self.use_score_multiset_permutation = bool(
            cfg.get("SCORE_MULTISET_PERMUTATION", False)
        )
        self.permutation_max_utility_delta = float(
            cfg.get("PERMUTATION_MAX_UTILITY_DELTA", 6.0)
        )
        self.permutation_use_acceptance = bool(
            cfg.get("PERMUTATION_USE_ACCEPTANCE", True)
        )
        self.permutation_acceptance_threshold = float(
            cfg.get("PERMUTATION_ACCEPTANCE_THRESHOLD", 0.0)
        )
        if self.use_score_multiset_permutation:
            permutation_heads = int(
                cfg.get("PERMUTATION_NUM_HEADS", num_heads)
            )
            if self.hidden_dim % permutation_heads != 0:
                raise ValueError(
                    "DEPLOYED_SET_SCORER hidden dimension must be divisible "
                    "by PERMUTATION_NUM_HEADS"
                )
            self.permutation_input_norm = nn.LayerNorm(self.hidden_dim)
            permutation_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=permutation_heads,
                dim_feedforward=(
                    self.hidden_dim
                    * int(cfg.get("PERMUTATION_FFN_MULTIPLIER", 3))
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.permutation_set_encoder = nn.TransformerEncoder(
                permutation_layer,
                num_layers=int(cfg.get("PERMUTATION_NUM_LAYERS", 2)),
                norm=nn.LayerNorm(self.hidden_dim),
            )
            self.permutation_type_heads = nn.ModuleList(
                [
                    _build_mlp(
                        self.hidden_dim,
                        self.hidden_dim,
                        1,
                        dropout,
                    )
                    for _ in range(3)
                ]
            )
            for head in self.permutation_type_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

            # Mean/max scene summaries plus four proposal-risk statistics:
            # learned/base top margins, residual magnitude and rank-change rate.
            self.permutation_acceptance_head = _build_mlp(
                self.hidden_dim * 2 + 4,
                self.hidden_dim,
                1,
                dropout,
            )
            nn.init.zeros_(self.permutation_acceptance_head[-1].weight)
            nn.init.constant_(
                self.permutation_acceptance_head[-1].bias,
                float(cfg.get("PERMUTATION_ACCEPTANCE_BIAS_INIT", -2.5)),
            )
        else:
            self.permutation_input_norm = None
            self.permutation_set_encoder = None
            self.permutation_type_heads = None
            self.permutation_acceptance_head = None

    @staticmethod
    def _normalize_score(score):
        centered = score - score.mean(dim=-1, keepdim=True)
        scale = centered.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        return centered / scale

    def _trajectory_features(self, trajectories):
        steps = self.measurement_steps.to(trajectories.device).clamp(
            0, trajectories.shape[3] - 1
        )
        previous_steps = (steps - 5).clamp_min(0)
        position = trajectories.index_select(3, steps).permute(
            0, 1, 3, 2, 4
        )
        previous = trajectories.index_select(3, previous_steps).permute(
            0, 1, 3, 2, 4
        )
        elapsed = (
            (steps - previous_steps).clamp_min(1).type_as(trajectories)
            * self.dt
        )
        velocity = (position - previous) / elapsed[
            None, None, :, None, None
        ]
        relative_position = position[..., 1, :] - position[..., 0, :]
        relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(0.5)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        return torch.cat(
            [
                position.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                relative_position / 20.0,
                relative_velocity / 20.0,
                distance / 20.0,
                closing_speed / 10.0,
                speed / 20.0,
            ],
            dim=-1,
        )

    def forward(
        self,
        world_hidden,
        trajectories,
        base_logits,
        selection_logits,
        branch_ids,
        pair_type_ids,
        scene_context,
        spatial_evidence,
    ):
        batch_size, num_modes = base_logits.shape
        expected_world = (
            batch_size,
            num_modes,
            self.num_horizons,
            self.hidden_dim,
        )
        if world_hidden.shape != expected_world:
            raise ValueError(
                f"Expected deployed world hidden {expected_world}, got "
                f"{tuple(world_hidden.shape)}"
            )
        expected_spatial = (
            batch_size,
            num_modes,
            self.num_horizons,
            SPATIAL_EVIDENCE_DIM,
        )
        if spatial_evidence.shape != expected_spatial:
            raise ValueError(
                f"Expected deployed spatial evidence {expected_spatial}, "
                f"got {tuple(spatial_evidence.shape)}"
            )
        score_features = torch.stack(
            [
                self._normalize_score(base_logits),
                self._normalize_score(selection_logits),
                branch_ids.type_as(base_logits),
            ],
            dim=-1,
        )
        pair_type_ids = pair_type_ids.to(base_logits.device).long().clamp(0, 2)
        hidden = (
            self.world_norm(world_hidden)
            + self.scene_norm(scene_context)[:, None, None]
            + self.trajectory_proj(
                self._trajectory_features(trajectories)
            )
            + self.spatial_proj(spatial_evidence.type_as(world_hidden))
            + self.score_proj(score_features)[:, :, None]
            + self.type_embedding(pair_type_ids)[:, None, None]
            + self.horizon_embedding[None, None]
        )
        hidden_dim = hidden.shape[-1]
        hidden = self.set_encoder(
            hidden.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_horizons,
                num_modes,
                hidden_dim,
            )
        ).reshape(
            batch_size, self.num_horizons, num_modes, hidden_dim
        ).permute(0, 2, 1, 3)
        hidden = self.horizon_encoder(
            hidden.reshape(
                batch_size * num_modes,
                self.num_horizons,
                hidden_dim,
            )
        ).reshape(
            batch_size, num_modes, self.num_horizons, hidden_dim
        )
        horizon_logits = self.utility_head(hidden).squeeze(-1)
        aggregate = (
            horizon_logits * self.horizon_weights.type_as(horizon_logits)[
                None, None
            ]
        ).sum(dim=-1)
        aggregate = aggregate - aggregate.mean(dim=-1, keepdim=True)
        residual = self.max_logit_delta * torch.tanh(
            aggregate / max(self.max_logit_delta, self.eps)
        )
        gate = torch.sigmoid(self.gate_logit)
        pre_permutation_logits = base_logits + gate * residual
        ret = {
            "joint_logits": pre_permutation_logits,
            "base_logits": base_logits,
            "horizon_logits": horizon_logits,
            "logit_residual": gate * residual,
            "gate": gate,
            "probability": torch.softmax(
                pre_permutation_logits, dim=-1
            ),
        }
        if not self.use_score_multiset_permutation:
            return ret

        mode_hidden = (
            hidden
            * self.horizon_weights.type_as(hidden)[None, None, :, None]
        ).sum(dim=2)
        permutation_hidden = self.permutation_set_encoder(
            self.permutation_input_norm(mode_hidden)
        )
        type_utility = torch.stack(
            [
                head(permutation_hidden).squeeze(-1)
                for head in self.permutation_type_heads
            ],
            dim=1,
        )
        batch_idx = torch.arange(
            batch_size, device=base_logits.device
        )
        raw_utility_delta = type_utility[batch_idx, pair_type_ids]
        raw_utility_delta = raw_utility_delta - raw_utility_delta.mean(
            dim=-1, keepdim=True
        )
        max_delta = max(self.permutation_max_utility_delta, self.eps)
        utility_delta = max_delta * torch.tanh(
            raw_utility_delta / max_delta
        )
        permutation_utility = (
            pre_permutation_logits.detach() + utility_delta
        )

        # order maps rank -> candidate. Scatter assigns rank-sorted original
        # scores back to candidate positions, preserving the score multiset
        # exactly. At zero initialization utility follows the original logits,
        # so this proposal is also an exact identity mapping.
        permutation_order = permutation_utility.argsort(
            dim=-1, descending=True
        )
        sorted_logits = pre_permutation_logits.sort(
            dim=-1, descending=True
        ).values
        proposal_logits = torch.empty_like(pre_permutation_logits).scatter(
            1, permutation_order, sorted_logits
        )
        base_order = pre_permutation_logits.detach().argsort(
            dim=-1, descending=True
        )
        rank_change_rate = (
            permutation_order != base_order
        ).type_as(base_logits).mean(dim=-1, keepdim=True)
        utility_top = permutation_utility.topk(
            k=min(2, num_modes), dim=-1
        ).values
        base_top = pre_permutation_logits.detach().topk(
            k=min(2, num_modes), dim=-1
        ).values
        if num_modes > 1:
            utility_margin = utility_top[:, :1] - utility_top[:, 1:2]
            base_margin = base_top[:, :1] - base_top[:, 1:2]
        else:
            utility_margin = utility_top.new_zeros(batch_size, 1)
            base_margin = base_top.new_zeros(batch_size, 1)
        acceptance_features = torch.cat(
            [
                permutation_hidden.mean(dim=1),
                permutation_hidden.amax(dim=1),
                utility_margin,
                base_margin,
                utility_delta.abs().mean(dim=-1, keepdim=True),
                rank_change_rate,
            ],
            dim=-1,
        )
        acceptance_logit = self.permutation_acceptance_head(
            acceptance_features
        ).squeeze(-1)
        if self.permutation_use_acceptance:
            permutation_accepted = (
                acceptance_logit >= self.permutation_acceptance_threshold
            )
        else:
            permutation_accepted = torch.ones_like(
                acceptance_logit, dtype=torch.bool
            )
        joint_logits = torch.where(
            permutation_accepted[:, None],
            proposal_logits,
            pre_permutation_logits,
        )
        ret.update(
            {
                "joint_logits": joint_logits,
                "probability": torch.softmax(joint_logits, dim=-1),
                "pre_permutation_logits": pre_permutation_logits,
                "permutation_proposal_logits": proposal_logits,
                "permutation_utility": permutation_utility,
                "permutation_utility_delta": utility_delta,
                "permutation_order": permutation_order,
                "permutation_acceptance_logit": acceptance_logit,
                "permutation_acceptance_probability": torch.sigmoid(
                    acceptance_logit
                ),
                "permutation_accepted": permutation_accepted,
                "permutation_rank_change_rate": rank_change_rate.squeeze(-1),
            }
        )
        return ret


class ConfidencePreservingDeployedTrajectoryRefiner(nn.Module):
    """Refine the frozen deployed set without changing confidence or membership.

    The trusted selector first fixes the six trajectories and their logits.  This
    module then predicts bounded, world-conditioned waypoint residuals for those
    trajectories only.  Its residual head is zero initialized, so adding the
    module is an exact functional no-op before training.
    """

    def __init__(
        self,
        cfg,
        hidden_dim,
        num_heads,
        measurement_steps,
        num_future_frames,
        dt,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = len(measurement_steps)
        self.num_future_frames = int(num_future_frames)
        self.dt = float(dt)
        self.eps = 1e-6
        if self.num_horizons != 3:
            raise ValueError(
                "Deployed trajectory refinement expects 3/5/8-second "
                "measurement steps"
            )

        dropout = float(cfg.get("DROPOUT", 0.1))
        steps = torch.as_tensor(measurement_steps, dtype=torch.long)
        if (
            steps.ndim != 1
            or steps.numel() != self.num_horizons
            or not bool((steps[1:] > steps[:-1]).all())
            or int(steps[-1]) >= self.num_future_frames
        ):
            raise ValueError("Invalid deployed-refiner measurement steps")
        self.register_buffer(
            "measurement_steps", steps, persistent=False
        )
        self.register_buffer(
            "interpolation_basis",
            self._build_interpolation_basis(
                steps, self.num_future_frames
            ),
            persistent=False,
        )

        max_deltas = torch.as_tensor(
            cfg.get("MAX_WAYPOINT_DELTAS", [0.35, 0.75, 1.50]),
            dtype=torch.float32,
        )
        if max_deltas.numel() != self.num_horizons:
            raise ValueError(
                "DEPLOYED_TRAJECTORY_REFINER.MAX_WAYPOINT_DELTAS must "
                "contain three values"
            )
        self.register_buffer(
            "max_waypoint_deltas", max_deltas, persistent=False
        )

        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [0.20, 0.30, 0.50]),
            dtype=torch.float32,
        )
        if (
            horizon_weights.numel() != self.num_horizons
            or not bool((horizon_weights >= 0).all())
            or float(horizon_weights.sum()) <= 0.0
        ):
            raise ValueError("Invalid deployed-refiner horizon weights")
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum(),
            persistent=False,
        )

        self.world_norm = nn.LayerNorm(self.hidden_dim)
        self.scene_norm = nn.LayerNorm(self.hidden_dim)
        self.trajectory_proj = _build_mlp(
            16, self.hidden_dim, self.hidden_dim, dropout
        )
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(SPATIAL_EVIDENCE_DIM),
            _build_mlp(
                SPATIAL_EVIDENCE_DIM,
                self.hidden_dim,
                self.hidden_dim,
                dropout,
            ),
        )
        self.score_proj = _build_mlp(
            3, self.hidden_dim, self.hidden_dim, dropout
        )
        self.type_embedding = nn.Embedding(3, self.hidden_dim)
        self.branch_embedding = nn.Embedding(2, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.num_horizons, self.hidden_dim)
        )
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.branch_embedding.weight)
        nn.init.normal_(self.horizon_embedding, std=0.02)

        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=(
                self.hidden_dim * int(cfg.get("FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(cfg.get("NUM_SET_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(cfg.get("NUM_HEADS", num_heads)),
            dim_feedforward=(
                self.hidden_dim * int(cfg.get("FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(cfg.get("NUM_TEMPORAL_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.residual_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            4,
            dropout,
        )
        self.gate_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            dropout,
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(
            self.gate_head[-1].bias,
            float(cfg.get("GATE_BIAS_INIT", -1.5)),
        )

    @staticmethod
    def _build_interpolation_basis(steps, num_future_frames):
        basis = torch.zeros(
            num_future_frames, steps.numel(), dtype=torch.float32
        )
        step_values = [int(value) for value in steps.tolist()]
        for frame in range(num_future_frames):
            if frame <= step_values[0]:
                basis[frame, 0] = (frame + 1) / (step_values[0] + 1)
                continue
            assigned = False
            for horizon in range(1, len(step_values)):
                if frame <= step_values[horizon]:
                    denominator = max(
                        step_values[horizon]
                        - step_values[horizon - 1],
                        1,
                    )
                    alpha = (
                        frame - step_values[horizon - 1]
                    ) / denominator
                    basis[frame, horizon - 1] = 1.0 - alpha
                    basis[frame, horizon] = alpha
                    assigned = True
                    break
            if not assigned:
                basis[frame, -1] = 1.0
        return basis

    @staticmethod
    def _normalize_score(score):
        centered = score - score.mean(dim=-1, keepdim=True)
        scale = centered.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        return centered / scale

    def _trajectory_features(self, trajectories):
        steps = self.measurement_steps.to(trajectories.device)
        previous_steps = (steps - 5).clamp_min(0)
        position = trajectories.index_select(3, steps).permute(
            0, 1, 3, 2, 4
        )
        previous = trajectories.index_select(3, previous_steps).permute(
            0, 1, 3, 2, 4
        )
        elapsed = (
            (steps - previous_steps).clamp_min(1).type_as(trajectories)
            * self.dt
        )
        velocity = (position - previous) / elapsed[
            None, None, :, None, None
        ]
        relative_position = position[..., 1, :] - position[..., 0, :]
        relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(0.5)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        return torch.cat(
            [
                position.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                relative_position / 20.0,
                relative_velocity / 20.0,
                distance / 20.0,
                closing_speed / 10.0,
                speed / 20.0,
            ],
            dim=-1,
        )

    def forward(
        self,
        world_hidden,
        trajectories,
        logits,
        selection_logits,
        branch_ids,
        pair_type_ids,
        scene_context,
        spatial_evidence,
    ):
        batch_size, num_modes = logits.shape
        expected_world = (
            batch_size,
            num_modes,
            self.num_horizons,
            self.hidden_dim,
        )
        if world_hidden.shape != expected_world:
            raise ValueError(
                f"Expected deployed world hidden {expected_world}, got "
                f"{tuple(world_hidden.shape)}"
            )
        score_features = torch.stack(
            [
                self._normalize_score(logits),
                self._normalize_score(selection_logits),
                branch_ids.type_as(logits),
            ],
            dim=-1,
        )
        pair_type_ids = pair_type_ids.to(logits.device).long().clamp(0, 2)
        hidden = (
            self.world_norm(world_hidden)
            + self.scene_norm(scene_context)[:, None, None]
            + self.trajectory_proj(
                self._trajectory_features(trajectories)
            )
            + self.spatial_proj(spatial_evidence.type_as(world_hidden))
            + self.score_proj(score_features)[:, :, None]
            + self.type_embedding(pair_type_ids)[:, None, None]
            + self.branch_embedding(branch_ids.long().clamp(0, 1))[
                :, :, None
            ]
            + self.horizon_embedding[None, None]
        )
        hidden = self.set_encoder(
            hidden.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_horizons,
                num_modes,
                self.hidden_dim,
            )
        ).reshape(
            batch_size,
            self.num_horizons,
            num_modes,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)
        hidden = self.temporal_encoder(
            hidden.reshape(
                batch_size * num_modes,
                self.num_horizons,
                self.hidden_dim,
            )
        ).reshape(
            batch_size,
            num_modes,
            self.num_horizons,
            self.hidden_dim,
        )

        raw_waypoint_delta = self.residual_head(hidden).reshape(
            batch_size, num_modes, self.num_horizons, 2, 2
        )
        gate = torch.sigmoid(self.gate_head(hidden)).unsqueeze(-1)
        waypoint_delta = (
            torch.tanh(raw_waypoint_delta)
            * self.max_waypoint_deltas.type_as(raw_waypoint_delta)[
                None, None, :, None, None
            ]
            * gate
        )
        dense_delta = torch.einsum(
            "th,bmhad->bmatd",
            self.interpolation_basis.type_as(waypoint_delta),
            waypoint_delta,
        )
        refined = trajectories + dense_delta
        return {
            "trajectories": refined,
            "base_trajectories": trajectories,
            "waypoint_delta": waypoint_delta,
            "dense_delta": dense_delta,
            "gate": gate.squeeze(-1),
            "hidden": hidden,
        }


class CandidateBankProtectedExpansion(nn.Module):
    """Generate complementary futures without writing into the base six.

    Every expansion starts from a frozen base trajectory and a distinct joint
    intention from the 8x8 candidate bank.  The endpoint warp gives the branch
    a meaningful multimodal prior, while world-conditioned horizon tokens
    learn a bounded correction and an admission probability.
    """

    def __init__(
        self,
        cfg,
        hidden_dim,
        num_expansion_modes,
        measurement_steps,
        num_future_frames,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_expansion_modes = int(num_expansion_modes)
        self.num_horizons = len(measurement_steps)
        self.num_future_frames = int(num_future_frames)
        self.num_heads = int(cfg.get("NUM_HEADS", 8))
        self.dropout = float(cfg.get("DROPOUT", 0.1))
        self.max_score_delta = float(cfg.get("MAX_SCORE_DELTA", 0.75))
        self.full_bank_decoding = bool(
            cfg.get("FULL_BANK_DECODING", False)
        )
        self.use_dense_temporal_refiner = bool(
            cfg.get("DENSE_TEMPORAL_REFINER", False)
        )
        self.num_dense_knots = int(cfg.get("NUM_DENSE_KNOTS", 8))
        self.pair_prior_score_weight = float(
            cfg.get("PAIR_PRIOR_SCORE_WEIGHT", 0.0)
        )
        self.use_factorized_marginal_donors = bool(
            cfg.get("FACTORIZED_MARGINAL_DONORS", False)
        )
        self.use_trajectory_temporal_scorer = bool(
            cfg.get("TRAJECTORY_TEMPORAL_SCORER", False)
        )
        self.use_set_marginal_gain_target = bool(
            cfg.get("SET_MARGINAL_GAIN_TARGET", False)
        )
        self.use_hierarchical_set_selector = bool(
            cfg.get("HIERARCHICAL_SET_SELECTOR", False)
        )
        self.use_unified_deployed_confidence = bool(
            cfg.get("UNIFIED_DEPLOYED_CONFIDENCE", False)
        )
        self.set_gain_score_scale = float(
            cfg.get("SET_GAIN_SCORE_SCALE", 1.0)
        )
        self.hierarchical_selection_score_scale = float(
            cfg.get("HIERARCHICAL_SELECTION_SCORE_SCALE", 0.10)
        )
        self.hierarchical_selection_offset = float(
            cfg.get("HIERARCHICAL_SELECTION_OFFSET", 1e-3)
        )
        self.hierarchical_confidence_offset = float(
            cfg.get("HIERARCHICAL_CONFIDENCE_OFFSET", -0.05)
        )
        self.hierarchical_confidence_score_scale = float(
            cfg.get("HIERARCHICAL_CONFIDENCE_SCORE_SCALE", 0.05)
        )
        self.num_trajectory_knots = int(
            cfg.get("NUM_TRAJECTORY_KNOTS", 8)
        )
        self.use_decoupled_confidence = bool(
            cfg.get("DECOUPLED_CONFIDENCE", False)
        )
        self.cross_score_residual_from_selection = bool(
            cfg.get("CROSS_SCORE_RESIDUAL_FROM_SELECTION", False)
        )
        self.max_cross_score_delta = float(
            cfg.get("MAX_CROSS_SCORE_DELTA", 3.0)
        )
        self.use_learned_pair_selector = bool(
            cfg.get("LEARNED_PAIR_SELECTOR", False)
        )
        self.pair_selector_max_delta = float(
            cfg.get("PAIR_SELECTOR_MAX_DELTA", 3.0)
        )
        self.pair_selector_direct_utility = bool(
            cfg.get("PAIR_SELECTOR_DIRECT_UTILITY", False)
        )
        self.pair_selector_prior_weight = float(
            cfg.get("PAIR_SELECTOR_PRIOR_WEIGHT", 1.0)
        )
        self.pair_selector_novelty_weight = float(
            cfg.get("PAIR_SELECTOR_NOVELTY_WEIGHT", 1.0)
        )
        self.use_pair_temporal_evolution = bool(
            cfg.get("LEARNED_PAIR_TEMPORAL_EVOLUTION", False)
        )
        self.use_pair_prototype_as_donor = bool(
            cfg.get("USE_PAIR_PROTOTYPE_AS_DONOR", False)
        )
        self.endpoint_consistent_trajectories = bool(
            cfg.get("ENDPOINT_CONSISTENT_TRAJECTORIES", False)
        )
        self.allow_base_pair_reuse = bool(
            cfg.get("ALLOW_BASE_PAIR_REUSE", False)
        )
        self.competitive_replacement_scoring = bool(
            cfg.get("COMPETITIVE_REPLACEMENT_SCORING", False)
        )
        self.competitive_utility_scale = float(
            cfg.get("COMPETITIVE_UTILITY_SCALE", 2.0)
        )
        self.competitive_admission_temperature = max(
            float(cfg.get("COMPETITIVE_ADMISSION_TEMPERATURE", 1.0)),
            1e-3,
        )
        self.eps = 1e-6

        if self.use_set_marginal_gain_target and int(
            cfg.get("MAX_REPLACEMENTS", 1)
        ) != 1:
            raise ValueError(
                "SET_MARGINAL_GAIN_TARGET currently requires "
                "MAX_REPLACEMENTS=1 so its train and deployment sets match"
            )
        if (
            self.use_hierarchical_set_selector
            and not self.use_set_marginal_gain_target
        ):
            raise ValueError(
                "HIERARCHICAL_SET_SELECTOR requires SET_MARGINAL_GAIN_TARGET"
            )
        if (
            self.use_unified_deployed_confidence
            and not self.use_trajectory_temporal_scorer
        ):
            raise ValueError(
                "UNIFIED_DEPLOYED_CONFIDENCE requires "
                "TRAJECTORY_TEMPORAL_SCORER"
            )

        if self.num_horizons != 3:
            raise ValueError(
                "Candidate-bank expansion expects the 3/5/8-second horizons"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "Expansion hidden dimension must be divisible by NUM_HEADS"
            )
        self.register_buffer(
            "measurement_steps",
            torch.as_tensor(measurement_steps, dtype=torch.long),
            persistent=False,
        )
        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [0.2, 0.3, 0.5]),
            dtype=torch.float32,
        )
        if horizon_weights.numel() != self.num_horizons:
            raise ValueError("Expansion HORIZON_WEIGHTS must contain 3 values")
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum().clamp_min(self.eps),
            persistent=False,
        )
        max_waypoint_deltas = torch.as_tensor(
            cfg.get("MAX_WAYPOINT_DELTAS", [0.5, 1.0, 2.0]),
            dtype=torch.float32,
        )
        if max_waypoint_deltas.numel() != self.num_horizons:
            raise ValueError(
                "Expansion MAX_WAYPOINT_DELTAS must contain 3 values"
            )
        self.register_buffer(
            "max_waypoint_deltas",
            max_waypoint_deltas,
            persistent=False,
        )
        pair_temporal_max_deltas = torch.as_tensor(
            cfg.get("PAIR_TEMPORAL_MAX_WAYPOINT_DELTAS", [1.0, 1.5, 0.0]),
            dtype=torch.float32,
        )
        if pair_temporal_max_deltas.numel() != self.num_horizons:
            raise ValueError(
                "PAIR_TEMPORAL_MAX_WAYPOINT_DELTAS must contain 3 values"
            )
        self.register_buffer(
            "pair_temporal_max_waypoint_deltas",
            pair_temporal_max_deltas,
            persistent=False,
        )
        if self.use_dense_temporal_refiner:
            if self.num_dense_knots < 2:
                raise ValueError(
                    "DENSE_TEMPORAL_REFINER requires at least 2 knots"
                )
            dense_max_deltas = torch.as_tensor(
                cfg.get(
                    "DENSE_TEMPORAL_MAX_DELTAS",
                    [0.25, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60],
                ),
                dtype=torch.float32,
            )
            if dense_max_deltas.numel() != self.num_dense_knots:
                raise ValueError(
                    "DENSE_TEMPORAL_MAX_DELTAS must match NUM_DENSE_KNOTS"
                )
            self.register_buffer(
                "dense_temporal_max_deltas",
                dense_max_deltas,
                persistent=False,
            )
            self.register_buffer(
                "dense_knot_steps",
                torch.linspace(
                    self.num_future_frames / self.num_dense_knots,
                    self.num_future_frames,
                    self.num_dense_knots,
                ),
                persistent=False,
            )
        else:
            self.register_buffer(
                "dense_temporal_max_deltas",
                torch.empty(0),
                persistent=False,
            )
            self.register_buffer(
                "dense_knot_steps",
                torch.empty(0),
                persistent=False,
            )
        if self.use_trajectory_temporal_scorer:
            if self.num_trajectory_knots < 3:
                raise ValueError(
                    "TRAJECTORY_TEMPORAL_SCORER requires at least 3 knots"
                )
            trajectory_knot_indices = torch.linspace(
                0,
                self.num_future_frames - 1,
                self.num_trajectory_knots,
            ).round().long().unique(sorted=True)
            if trajectory_knot_indices.numel() != self.num_trajectory_knots:
                raise ValueError(
                    "NUM_TRAJECTORY_KNOTS creates duplicate frame indices"
                )
            self.register_buffer(
                "trajectory_knot_indices",
                trajectory_knot_indices,
                persistent=False,
            )
        else:
            self.register_buffer(
                "trajectory_knot_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

        self.pair_proj = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.donor_proj = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.endpoint_proj = _build_mlp(
            4,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.score_proj = _build_mlp(
            1,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.mode_embedding = nn.Parameter(
            torch.empty(self.num_expansion_modes, self.hidden_dim)
        )
        self.horizon_embedding = nn.Parameter(
            torch.empty(self.num_horizons, self.hidden_dim)
        )
        nn.init.normal_(self.mode_embedding, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)

        self.base_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.hidden_dim)
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim
            * int(cfg.get("FFN_MULTIPLIER", 2)),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(cfg.get("NUM_SET_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        horizon_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim
            * int(cfg.get("FFN_MULTIPLIER", 2)),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.horizon_encoder = nn.TransformerEncoder(
            horizon_layer,
            num_layers=int(cfg.get("NUM_HORIZON_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        if self.use_dense_temporal_refiner:
            self.dense_knot_embedding = nn.Parameter(
                torch.empty(self.num_dense_knots, self.hidden_dim)
            )
            nn.init.normal_(self.dense_knot_embedding, std=0.02)
            self.dense_horizon_proj = nn.Linear(
                self.hidden_dim, self.hidden_dim
            )
            dense_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim
                * int(cfg.get("DENSE_FFN_MULTIPLIER", 2)),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.dense_temporal_encoder = nn.TransformerEncoder(
                dense_layer,
                num_layers=int(cfg.get("NUM_DENSE_TEMPORAL_LAYERS", 1)),
                norm=nn.LayerNorm(self.hidden_dim),
            )
            self.dense_temporal_head = nn.Linear(self.hidden_dim, 2 * 2)
            nn.init.zeros_(self.dense_temporal_head.weight)
            nn.init.zeros_(self.dense_temporal_head.bias)
        else:
            self.register_parameter("dense_knot_embedding", None)
            self.dense_horizon_proj = None
            self.dense_temporal_encoder = None
            self.dense_temporal_head = None
        if self.use_trajectory_temporal_scorer:
            # Per knot: two-agent position, velocity and acceleration (12),
            # relative position/velocity (4), distance (1), progress (1).
            trajectory_feature_dim = 18
            self.trajectory_geometry_proj = _build_mlp(
                trajectory_feature_dim,
                self.hidden_dim,
                self.hidden_dim,
                self.dropout,
            )
            self.trajectory_knot_embedding = nn.Parameter(
                torch.empty(self.num_trajectory_knots, self.hidden_dim)
            )
            nn.init.normal_(self.trajectory_knot_embedding, std=0.02)
            trajectory_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim
                * int(cfg.get("TRAJECTORY_FFN_MULTIPLIER", 2)),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.trajectory_temporal_encoder = nn.TransformerEncoder(
                trajectory_layer,
                num_layers=int(cfg.get("NUM_TRAJECTORY_LAYERS", 2)),
                norm=nn.LayerNorm(self.hidden_dim),
            )
            self.trajectory_horizon_proj = nn.Linear(
                self.hidden_dim, self.hidden_dim
            )
        else:
            self.trajectory_geometry_proj = None
            self.register_parameter("trajectory_knot_embedding", None)
            self.trajectory_temporal_encoder = None
            self.trajectory_horizon_proj = None
        self.summary = _build_mlp(
            self.hidden_dim * 2,
            self.hidden_dim * 2,
            self.hidden_dim,
            self.dropout,
        )
        self.waypoint_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim * 2,
            2 * self.num_horizons * 2,
            self.dropout,
        )
        self.goal_gate_head = nn.Linear(self.hidden_dim, 1)
        self.score_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.dropout,
        )
        self.horizon_score_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.dropout,
        )
        self.admission_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.dropout,
        )
        if self.use_hierarchical_set_selector:
            # Replacement is sparse at scene level.  Predicting it separately
            # prevents the 64-way candidate ranking signal from being drowned
            # by scenes whose protected set already covers the ground truth.
            self.scene_replacement_gate_head = _build_mlp(
                self.hidden_dim * 3 + 4,
                self.hidden_dim * 2,
                1,
                self.dropout,
            )
            nn.init.normal_(
                self.scene_replacement_gate_head[-1].weight, std=1e-3
            )
            nn.init.constant_(
                self.scene_replacement_gate_head[-1].bias,
                float(cfg.get("SCENE_REPLACEMENT_GATE_BIAS", -2.0)),
            )
        else:
            self.scene_replacement_gate_head = None

        if self.use_unified_deployed_confidence:
            confidence_hidden = int(
                cfg.get("UNIFIED_CONFIDENCE_HIDDEN_DIM", self.hidden_dim)
            )
            self.base_confidence_proj = _build_mlp(
                self.hidden_dim * 2,
                confidence_hidden,
                self.hidden_dim,
                self.dropout,
            )
            self.expansion_confidence_proj = _build_mlp(
                self.hidden_dim * 2,
                confidence_hidden,
                self.hidden_dim,
                self.dropout,
            )
            self.confidence_prior_proj = _build_mlp(
                3,
                max(self.hidden_dim // 2, 32),
                self.hidden_dim,
                self.dropout,
            )
            confidence_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim
                * int(cfg.get("UNIFIED_CONFIDENCE_FFN_MULTIPLIER", 2)),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.confidence_set_encoder = nn.TransformerEncoder(
                confidence_layer,
                num_layers=int(
                    cfg.get("NUM_UNIFIED_CONFIDENCE_LAYERS", 2)
                ),
                norm=nn.LayerNorm(self.hidden_dim),
            )
            self.confidence_residual_head = _build_mlp(
                self.hidden_dim,
                confidence_hidden,
                1,
                self.dropout,
            )
            self.confidence_temperature_head = _build_mlp(
                self.hidden_dim * 2,
                confidence_hidden,
                1,
                self.dropout,
            )
            nn.init.zeros_(self.confidence_residual_head[-1].weight)
            nn.init.zeros_(self.confidence_residual_head[-1].bias)
            nn.init.zeros_(self.confidence_temperature_head[-1].weight)
            nn.init.zeros_(self.confidence_temperature_head[-1].bias)
            self.unified_confidence_gate_logit = nn.Parameter(
                torch.tensor(
                    float(cfg.get("UNIFIED_CONFIDENCE_GATE_BIAS", -1.0))
                )
            )
            self.max_unified_confidence_delta = float(
                cfg.get("MAX_UNIFIED_CONFIDENCE_DELTA", 2.0)
            )
            self.max_unified_log_temperature = math.log(
                max(
                    float(cfg.get("MAX_UNIFIED_CONFIDENCE_TEMPERATURE", 2.0)),
                    1.0,
                )
            )
        else:
            self.base_confidence_proj = None
            self.expansion_confidence_proj = None
            self.confidence_prior_proj = None
            self.confidence_set_encoder = None
            self.confidence_residual_head = None
            self.confidence_temperature_head = None
            self.register_parameter("unified_confidence_gate_logit", None)
            self.max_unified_confidence_delta = 0.0
            self.max_unified_log_temperature = 0.0
        if self.competitive_replacement_scoring:
            # Predict one cross-set utility used by both admission and
            # replacement ordering.  Sharing this value prevents the two
            # decisions from learning contradictory policies.
            self.replacement_utility_head = _build_mlp(
                self.hidden_dim + 3,
                self.hidden_dim,
                1,
                self.dropout,
            )
            nn.init.zeros_(self.replacement_utility_head[-1].weight)
            nn.init.zeros_(self.replacement_utility_head[-1].bias)
        else:
            self.replacement_utility_head = None
        if self.use_decoupled_confidence:
            self.cross_branch_score_head = _build_mlp(
                self.hidden_dim + 4,
                int(cfg.get("CROSS_SCORE_HIDDEN_DIM", self.hidden_dim)),
                1,
                self.dropout,
            )
            self.cross_branch_score_bias = nn.Parameter(
                torch.tensor(
                    float(cfg.get("CROSS_SCORE_BIAS_INIT", -1.0))
                )
            )
            self.cross_branch_score_gate_logit = nn.Parameter(
                torch.tensor(
                    float(cfg.get("CROSS_SCORE_GATE_BIAS_INIT", -3.0))
                )
            )
            nn.init.zeros_(self.cross_branch_score_head[-1].weight)
            nn.init.zeros_(self.cross_branch_score_head[-1].bias)
        else:
            self.cross_branch_score_head = None
            self.register_parameter("cross_branch_score_bias", None)
            self.register_parameter("cross_branch_score_gate_logit", None)
        self.score_bias = nn.Parameter(
            torch.full(
                (self.num_expansion_modes,),
                float(cfg.get("INITIAL_SCORE_BIAS", -4.0)),
            )
        )
        if self.use_learned_pair_selector:
            self.pair_selector_pair_proj = _build_mlp(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                self.dropout,
            )
            self.pair_selector_geometry_proj = _build_mlp(
                7,
                self.hidden_dim,
                self.hidden_dim,
                self.dropout,
            )
            self.pair_selector_cross_attention = nn.MultiheadAttention(
                self.hidden_dim,
                self.num_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.pair_selector_norm = nn.LayerNorm(self.hidden_dim)
            self.pair_selector_horizon_head = nn.Linear(
                self.hidden_dim, self.num_horizons
            )
            self.pair_selector_rescue_head = nn.Linear(
                self.hidden_dim, self.num_horizons
            )
            self.pair_selector_quality_head = nn.Linear(self.hidden_dim, 1)
            self.pair_selector_temporal_head = (
                nn.Linear(
                    self.hidden_dim,
                    2 * self.num_horizons * 2,
                )
                if self.use_pair_temporal_evolution
                else None
            )
            self.pair_selector_gate_logit = nn.Parameter(
                torch.tensor(
                    float(cfg.get("PAIR_SELECTOR_GATE_BIAS_INIT", -3.0))
                )
            )
            selector_heads = [
                self.pair_selector_horizon_head,
                self.pair_selector_rescue_head,
                self.pair_selector_quality_head,
            ]
            if self.pair_selector_temporal_head is not None:
                selector_heads.append(self.pair_selector_temporal_head)
            for head in selector_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            self.pair_selector_pair_proj = None
            self.pair_selector_geometry_proj = None
            self.pair_selector_cross_attention = None
            self.pair_selector_norm = None
            self.pair_selector_horizon_head = None
            self.pair_selector_rescue_head = None
            self.pair_selector_quality_head = None
            self.pair_selector_temporal_head = None
            self.register_parameter("pair_selector_gate_logit", None)

        nn.init.zeros_(self.waypoint_head[-1].weight)
        nn.init.zeros_(self.waypoint_head[-1].bias)
        nn.init.zeros_(self.goal_gate_head.weight)
        nn.init.constant_(
            self.goal_gate_head.bias,
            float(cfg.get("GOAL_GATE_BIAS_INIT", -2.0)),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.zeros_(self.horizon_score_head[-1].weight)
        nn.init.zeros_(self.horizon_score_head[-1].bias)
        nn.init.normal_(self.admission_head[-1].weight, std=1e-3)
        nn.init.constant_(
            self.admission_head[-1].bias,
            float(cfg.get("ADMISSION_BIAS", -5.0)),
        )

    def score_pair_bank(
        self,
        pair_hidden,
        pair_endpoints,
        pair_logits,
        base_hidden,
        base_logits,
        base_trajs,
    ):
        """Predict complementary utility for every 8x8 joint intention."""
        batch_size, num_pairs = pair_logits.shape
        pair_endpoint = pair_endpoints.reshape(batch_size, num_pairs, -1)
        base_endpoint = base_trajs[..., -1, :].reshape(
            batch_size, base_trajs.shape[1], -1
        )
        endpoint_distance = torch.cdist(
            pair_endpoint.float(), base_endpoint.float()
        ).type_as(pair_logits)
        donor_indices = endpoint_distance.argmin(dim=-1)
        donor_trajs = base_trajs.gather(
            1,
            donor_indices[..., None, None, None].expand(
                -1,
                -1,
                base_trajs.shape[2],
                base_trajs.shape[3],
                base_trajs.shape[4],
            ),
        )
        factorized_donor_indices = donor_indices[..., None].expand(
            -1, -1, base_trajs.shape[2]
        )
        prototype_donor_trajs = donor_trajs
        if self.use_factorized_marginal_donors:
            # A joint nearest neighbour can force both agents to inherit the
            # same base mode even when the bank combines two different
            # marginal intentions. Select each agent's motion primitive
            # independently, then compose the pair before interaction
            # refinement.
            base_agent_endpoints = base_trajs[..., -1, :].permute(
                0, 2, 1, 3
            )
            marginal_distance = torch.linalg.vector_norm(
                pair_endpoints[:, :, :, None]
                - base_agent_endpoints[:, None],
                dim=-1,
            )
            factorized_donor_indices = marginal_distance.argmin(dim=-1)
            base_agent_trajs = base_trajs.permute(0, 2, 1, 3, 4)
            gather_indices = factorized_donor_indices.permute(
                0, 2, 1
            )[..., None, None].expand(
                -1,
                -1,
                -1,
                base_trajs.shape[3],
                base_trajs.shape[4],
            )
            prototype_donor_trajs = base_agent_trajs.gather(
                2, gather_indices
            ).permute(0, 2, 1, 3, 4)
        donor_logits = base_logits.gather(1, donor_indices)
        endpoint_delta = (
            pair_endpoints - prototype_donor_trajs[..., -1, :]
        )
        pair_score = pair_logits - pair_logits.mean(dim=-1, keepdim=True)
        pair_score = pair_score / pair_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        base_score = donor_logits - base_logits.mean(dim=-1, keepdim=True)
        base_score = base_score / base_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        nearest_base = endpoint_distance.amin(dim=-1)

        progress = torch.linspace(
            1.0 / self.num_future_frames,
            1.0,
            self.num_future_frames,
            device=base_trajs.device,
            dtype=base_trajs.dtype,
        )
        smooth_progress = progress.square() * (3.0 - 2.0 * progress)
        prototype_trajs = prototype_donor_trajs + smooth_progress[
            None, None, None, :, None
        ] * endpoint_delta[:, :, :, None]

        if not self.use_learned_pair_selector:
            zeros_horizon = pair_logits.new_zeros(
                batch_size, num_pairs, self.num_horizons
            )
            return {
                "horizon_logits": zeros_horizon,
                "rescue_logits": zeros_horizon,
                "quality_logits": pair_logits.new_zeros(batch_size, num_pairs),
                "selection_delta": pair_logits.new_zeros(batch_size, num_pairs),
                "gate": pair_logits.new_zeros(()),
                "donor_indices": donor_indices,
                "factorized_donor_indices": factorized_donor_indices,
                "prototype_trajs": prototype_trajs,
                "temporal_waypoint_delta": pair_logits.new_zeros(
                    batch_size,
                    num_pairs,
                    2,
                    self.num_horizons,
                    2,
                ),
            }

        geometry = torch.cat(
            [
                endpoint_delta.flatten(start_dim=-2) / 50.0,
                pair_score[..., None],
                base_score[..., None],
                torch.tanh(nearest_base[..., None] / 5.0),
            ],
            dim=-1,
        )
        token = self.pair_selector_pair_proj(pair_hidden)
        token = token + self.pair_selector_geometry_proj(geometry)
        context, _ = self.pair_selector_cross_attention(
            token, base_hidden, base_hidden, need_weights=False
        )
        token = self.pair_selector_norm(token + context)
        horizon_logits = self.pair_selector_horizon_head(token)
        rescue_logits = self.pair_selector_rescue_head(token)
        quality_logits = self.pair_selector_quality_head(token).squeeze(-1)
        temporal_waypoint_delta = pair_logits.new_zeros(
            batch_size,
            num_pairs,
            2,
            self.num_horizons,
            2,
        )
        if self.use_pair_temporal_evolution:
            temporal_waypoint_delta = self.pair_selector_temporal_head(
                token
            ).reshape(
                batch_size,
                num_pairs,
                2,
                self.num_horizons,
                2,
            )
            temporal_waypoint_delta = torch.tanh(
                temporal_waypoint_delta
            ) * self.pair_temporal_max_waypoint_deltas.type_as(
                temporal_waypoint_delta
            )[None, None, None, :, None]
            prototype_trajs = prototype_trajs + self._interpolate_waypoints(
                temporal_waypoint_delta
            )
        expected_utility = (
            torch.sigmoid(horizon_logits)
            + 2.0 * torch.sigmoid(rescue_logits)
        ).mul(self.horizon_weights[None, None]).sum(dim=-1)
        expected_utility = expected_utility + 0.25 * torch.tanh(
            quality_logits
        )
        expected_utility = expected_utility - expected_utility.mean(
            dim=-1, keepdim=True
        )
        learned_gate = torch.sigmoid(self.pair_selector_gate_logit)
        if self.pair_selector_direct_utility:
            gate = learned_gate + (1.0 - learned_gate).detach()
            selection_delta = (
                gate
                * self.pair_selector_max_delta
                * torch.tanh(expected_utility)
            )
        else:
            gate = learned_gate
            selection_delta = (
                gate
                * self.pair_selector_max_delta
                * torch.tanh(expected_utility)
            )
        return {
            "horizon_logits": horizon_logits,
            "rescue_logits": rescue_logits,
            "quality_logits": quality_logits,
            "selection_delta": selection_delta,
            "gate": gate,
            "donor_indices": donor_indices,
            "factorized_donor_indices": factorized_donor_indices,
            "prototype_trajs": prototype_trajs,
            "temporal_waypoint_delta": temporal_waypoint_delta,
        }

    def _trajectory_geometry_tokens(self, trajectories, pair_hidden):
        """Encode the complete candidate motion, not only its endpoint."""
        knot_indices = self.trajectory_knot_indices.to(
            trajectories.device
        )
        knots = trajectories.index_select(3, knot_indices).permute(
            0, 1, 3, 2, 4
        )
        knot_times = (knot_indices.type_as(knots) + 1.0) * 0.1
        interval = torch.diff(
            torch.cat([knot_times.new_zeros(1), knot_times])
        ).clamp_min(0.1)
        displacement = torch.diff(
            torch.cat([knots[:, :, :1], knots], dim=2),
            dim=2,
        )
        velocity = displacement / interval[None, None, :, None, None]
        velocity[:, :, 0] = velocity[:, :, 1]
        acceleration = torch.diff(
            torch.cat([velocity[:, :, :1], velocity], dim=2),
            dim=2,
        ) / interval[None, None, :, None, None]
        acceleration[:, :, 0] = acceleration[:, :, 1]

        relative_position = knots[:, :, :, 1] - knots[:, :, :, 0]
        relative_velocity = velocity[:, :, :, 1] - velocity[:, :, :, 0]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        progress = knot_times / knot_times[-1].clamp_min(self.eps)
        features = torch.cat(
            [
                knots.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                acceleration.flatten(start_dim=-2) / 10.0,
                relative_position / 50.0,
                relative_velocity / 20.0,
                distance / 20.0,
                progress[None, None, :, None].expand(
                    knots.shape[0], knots.shape[1], -1, -1
                ),
            ],
            dim=-1,
        )
        token = (
            self.trajectory_geometry_proj(features)
            + self.pair_proj(pair_hidden)[:, :, None]
            + self.trajectory_knot_embedding[None, None]
        )
        return self.trajectory_temporal_encoder(
            token.reshape(
                trajectories.shape[0] * trajectories.shape[1],
                self.num_trajectory_knots,
                self.hidden_dim,
            )
        ).reshape(
            trajectories.shape[0],
            trajectories.shape[1],
            self.num_trajectory_knots,
            self.hidden_dim,
        )

    def _unified_confidence_logits(
        self,
        base_trajs,
        base_hidden,
        base_logits,
        expansion_summary,
        expansion_trajectory_token,
        candidate_selector_logits,
    ):
        """Score protected and generated modes in one comparable set."""
        base_geometry = self._trajectory_geometry_tokens(
            base_trajs.detach(), base_hidden.detach()
        ).mean(dim=2)
        base_token = self.base_confidence_proj(
            torch.cat([base_hidden.detach(), base_geometry], dim=-1)
        )
        expansion_token = self.expansion_confidence_proj(
            torch.cat(
                [
                    expansion_summary,
                    expansion_trajectory_token.mean(dim=2),
                ],
                dim=-1,
            )
        )

        centered_selector = candidate_selector_logits - (
            candidate_selector_logits.mean(dim=-1, keepdim=True)
        )
        base_floor = base_logits.amin(dim=-1, keepdim=True)
        expansion_anchor = (
            base_floor
            + self.hierarchical_confidence_offset
            + self.hierarchical_confidence_score_scale
            * torch.tanh(centered_selector)
        )
        anchor_logits = torch.cat([base_logits, expansion_anchor], dim=-1)
        anchor_mean = anchor_logits.mean(dim=-1, keepdim=True)
        anchor_std = anchor_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        normalized_anchor = (anchor_logits - anchor_mean) / anchor_std
        branch_id = torch.cat(
            [
                torch.zeros_like(base_logits),
                torch.ones_like(expansion_anchor),
            ],
            dim=-1,
        )
        prior_feature = torch.stack(
            [normalized_anchor, torch.tanh(normalized_anchor), branch_id],
            dim=-1,
        )
        confidence_token = torch.cat([base_token, expansion_token], dim=1)
        confidence_token = confidence_token + self.confidence_prior_proj(
            prior_feature
        )
        confidence_token = self.confidence_set_encoder(confidence_token)
        confidence_residual = self.max_unified_confidence_delta * torch.tanh(
            self.confidence_residual_head(confidence_token).squeeze(-1)
        )
        scene_confidence = torch.cat(
            [
                confidence_token.mean(dim=1),
                confidence_token.amax(dim=1),
            ],
            dim=-1,
        )
        log_temperature = self.max_unified_log_temperature * torch.tanh(
            self.confidence_temperature_head(scene_confidence).squeeze(-1)
        )
        temperature = torch.exp(log_temperature)
        gate = torch.sigmoid(self.unified_confidence_gate_logit)
        calibrated_logits = (
            anchor_logits * temperature[:, None]
            + gate * confidence_residual
        )
        return {
            "base_logits": calibrated_logits[:, : base_logits.shape[1]],
            "expansion_logits": calibrated_logits[:, base_logits.shape[1] :],
            "anchor_logits": anchor_logits,
            "residual": confidence_residual,
            "temperature": temperature,
            "gate": gate,
        }

    def _interpolate_waypoints(self, waypoint_delta):
        """Piecewise-linearly interpolate 3/5/8-second residual knots."""
        batch_size, num_modes, num_agents = waypoint_delta.shape[:3]
        knot_times = torch.cat(
            [
                waypoint_delta.new_zeros(1),
                self.measurement_steps.to(
                    waypoint_delta.device
                ).type_as(waypoint_delta)
                + 1.0,
            ]
        )
        knot_values = torch.cat(
            [
                waypoint_delta.new_zeros(
                    batch_size, num_modes, num_agents, 1, 2
                ),
                waypoint_delta,
            ],
            dim=3,
        )
        times = torch.arange(
            1,
            self.num_future_frames + 1,
            device=waypoint_delta.device,
            dtype=waypoint_delta.dtype,
        )
        right = torch.bucketize(times, knot_times, right=False).clamp(
            1, self.num_horizons
        )
        left = right - 1
        alpha = (
            (times - knot_times[left])
            / (knot_times[right] - knot_times[left]).clamp_min(self.eps)
        )
        left_value = knot_values.index_select(3, left)
        right_value = knot_values.index_select(3, right)
        return left_value + alpha[None, None, None, :, None] * (
            right_value - left_value
        )

    def _interpolate_dense_knots(self, knot_delta):
        """Interpolate uniformly spaced residual knots to all future frames."""
        batch_size, num_modes, num_agents = knot_delta.shape[:3]
        knot_times = torch.cat(
            [
                knot_delta.new_zeros(1),
                self.dense_knot_steps.to(knot_delta.device).type_as(
                    knot_delta
                ),
            ]
        )
        knot_values = torch.cat(
            [
                knot_delta.new_zeros(
                    batch_size, num_modes, num_agents, 1, 2
                ),
                knot_delta,
            ],
            dim=3,
        )
        times = torch.arange(
            1,
            self.num_future_frames + 1,
            device=knot_delta.device,
            dtype=knot_delta.dtype,
        )
        right = torch.bucketize(times, knot_times, right=False).clamp(
            1, self.num_dense_knots
        )
        left = right - 1
        alpha = (
            (times - knot_times[left])
            / (knot_times[right] - knot_times[left]).clamp_min(self.eps)
        )
        left_value = knot_values.index_select(3, left)
        right_value = knot_values.index_select(3, right)
        return left_value + alpha[None, None, None, :, None] * (
            right_value - left_value
        )

    def forward(
        self,
        pair_hidden,
        pair_endpoints,
        donor_hidden,
        donor_logits,
        donor_trajs,
        proposal_trajs,
        proposal_prior,
        base_hidden,
        base_logits,
        base_trajs,
    ):
        batch_size, num_expansion, hidden_dim = pair_hidden.shape
        if (
            num_expansion != self.num_expansion_modes
            or hidden_dim != self.hidden_dim
        ):
            raise ValueError("Candidate expansion received an invalid shape")
        endpoint_delta = pair_endpoints - donor_trajs[..., -1, :]
        initial_trajs = (
            donor_trajs if proposal_trajs is None else proposal_trajs
        )
        donor_score = donor_logits - base_logits.mean(dim=-1, keepdim=True)
        donor_score = donor_score / base_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        token = (
            self.pair_proj(pair_hidden)
            + self.donor_proj(donor_hidden)
            + self.endpoint_proj(
                endpoint_delta.flatten(start_dim=-2) / 50.0
            )
            + self.score_proj(donor_score[..., None])
            + self.mode_embedding[None]
        )
        token = token[:, :, None] + self.horizon_embedding[None, None]

        query = token.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_horizons,
            num_expansion,
            self.hidden_dim,
        )
        base = base_hidden[:, None].expand(
            -1, self.num_horizons, -1, -1
        ).reshape(
            batch_size * self.num_horizons,
            base_hidden.shape[1],
            self.hidden_dim,
        )
        context, _ = self.base_cross_attention(
            query, base, base, need_weights=False
        )
        query = self.cross_norm(query + context)
        query = self.set_encoder(query)
        horizon_token = query.reshape(
            batch_size,
            self.num_horizons,
            num_expansion,
            self.hidden_dim,
        ).permute(0, 2, 1, 3)
        horizon_token = self.horizon_encoder(
            horizon_token.reshape(
                batch_size * num_expansion,
                self.num_horizons,
                self.hidden_dim,
            )
        ).reshape(
            batch_size,
            num_expansion,
            self.num_horizons,
            self.hidden_dim,
        )
        trajectory_token = None
        if self.use_trajectory_temporal_scorer:
            trajectory_token = self._trajectory_geometry_tokens(
                initial_trajs, pair_hidden
            )
            knot_times = (
                self.trajectory_knot_indices.to(horizon_token.device)
                .type_as(horizon_token)
                + 1.0
            )
            horizon_times = (
                self.measurement_steps.to(horizon_token.device)
                .type_as(horizon_token)
                + 1.0
            )
            trajectory_horizon_weight = torch.softmax(
                -(
                    horizon_times[:, None] - knot_times[None]
                ).abs()
                / max(float(self.num_future_frames) / 8.0, 1.0),
                dim=-1,
            )
            trajectory_horizon = torch.einsum(
                "hk,bmkd->bmhd",
                trajectory_horizon_weight,
                trajectory_token,
            )
            horizon_token = horizon_token + self.trajectory_horizon_proj(
                trajectory_horizon
            )
        weighted = (
            horizon_token
            * self.horizon_weights.type_as(horizon_token)[
                None, None, :, None
            ]
        ).sum(dim=2)
        summary = self.summary(
            torch.cat([weighted, horizon_token[:, :, -1]], dim=-1)
        )

        waypoint_delta = self.waypoint_head(summary).reshape(
            batch_size,
            num_expansion,
            2,
            self.num_horizons,
            2,
        )
        waypoint_delta = torch.tanh(waypoint_delta)
        waypoint_delta = (
            waypoint_delta
            * self.max_waypoint_deltas.type_as(waypoint_delta)[
                None, None, None, :, None
            ]
        )
        learned_delta = self._interpolate_waypoints(waypoint_delta)
        dense_knot_delta = None
        dense_temporal_delta = learned_delta.new_zeros(learned_delta.shape)
        if self.use_dense_temporal_refiner:
            horizon_steps = (
                self.measurement_steps.to(horizon_token.device)
                .type_as(horizon_token)
                + 1.0
            )
            knot_steps = self.dense_knot_steps.to(
                horizon_token.device
            ).type_as(horizon_token)
            horizon_weight = torch.softmax(
                -(
                    knot_steps[:, None] - horizon_steps[None]
                ).abs()
                / max(float(self.num_future_frames) / 8.0, 1.0),
                dim=-1,
            )
            dense_horizon = torch.einsum(
                "kh,bmhd->bmkd", horizon_weight, horizon_token
            )
            dense_token = (
                summary[:, :, None]
                + self.dense_horizon_proj(dense_horizon)
                + self.dense_knot_embedding[None, None]
            )
            dense_token = self.dense_temporal_encoder(
                dense_token.reshape(
                    batch_size * num_expansion,
                    self.num_dense_knots,
                    self.hidden_dim,
                )
            ).reshape(
                batch_size,
                num_expansion,
                self.num_dense_knots,
                self.hidden_dim,
            )
            dense_knot_delta = self.dense_temporal_head(
                dense_token
            ).reshape(
                batch_size,
                num_expansion,
                self.num_dense_knots,
                2,
                2,
            ).permute(0, 1, 3, 2, 4)
            dense_knot_delta = torch.tanh(dense_knot_delta)
            dense_knot_delta = (
                dense_knot_delta
                * self.dense_temporal_max_deltas.type_as(
                    dense_knot_delta
                )[None, None, None, :, None]
            )
            dense_temporal_delta = self._interpolate_dense_knots(
                dense_knot_delta
            )
        progress = torch.linspace(
            1.0 / self.num_future_frames,
            1.0,
            self.num_future_frames,
            device=initial_trajs.device,
            dtype=initial_trajs.dtype,
        )
        smooth_progress = progress.square() * (3.0 - 2.0 * progress)
        raw_goal_gate = torch.sigmoid(
            self.goal_gate_head(summary).squeeze(-1)
        )
        if self.endpoint_consistent_trajectories:
            # Preserve the selected joint intention exactly at 8 seconds while
            # letting the refiner reshape the intermediate future.
            learned_delta = learned_delta - smooth_progress[
                None, None, None, :, None
            ] * learned_delta[..., -1:, :]
            dense_temporal_delta = dense_temporal_delta - smooth_progress[
                None, None, None, :, None
            ] * dense_temporal_delta[..., -1:, :]
            goal_gate = raw_goal_gate + (1.0 - raw_goal_gate).detach()
            if proposal_trajs is not None:
                anchor_delta = initial_trajs.new_zeros(
                    initial_trajs.shape
                )
            else:
                anchor_delta = (
                    smooth_progress[None, None, None, :, None]
                    * endpoint_delta[:, :, :, None]
                )
        else:
            goal_gate = raw_goal_gate
            anchor_delta = (
                goal_gate[:, :, None, None, None]
                * smooth_progress[None, None, None, :, None]
                * endpoint_delta[:, :, :, None]
            )
        expansion_trajs = (
            initial_trajs.detach()
            + anchor_delta
            + learned_delta
            + dense_temporal_delta
        )

        horizon_logits = self.horizon_score_head(
            horizon_token
        ).squeeze(-1)
        legacy_admission_logits = self.admission_head(summary).squeeze(-1)
        selection_score_delta = self.max_score_delta * torch.tanh(
            self.score_head(summary).squeeze(-1)
        )
        legacy_selection_logits = (
            donor_logits.detach()
            + self.score_bias[None]
            + self.pair_prior_score_weight
            * (
                proposal_prior
                if proposal_prior is not None
                else donor_score.new_zeros(donor_score.shape)
            )
            + selection_score_delta
            + F.logsigmoid(legacy_admission_logits)
        )
        replacement_utility = legacy_admission_logits
        candidate_selector_logits = legacy_admission_logits
        scene_replacement_gate_logits = legacy_admission_logits.amax(dim=-1)
        if self.use_hierarchical_set_selector:
            base_logit_stats = torch.stack(
                [
                    base_logits.mean(dim=-1),
                    base_logits.std(dim=-1, unbiased=False),
                    base_logits.amax(dim=-1),
                    base_logits.amin(dim=-1),
                ],
                dim=-1,
            )
            scene_gate_context = torch.cat(
                [
                    summary.mean(dim=1),
                    summary.amax(dim=1),
                    base_hidden.detach().mean(dim=1),
                    base_logit_stats.detach(),
                ],
                dim=-1,
            )
            scene_replacement_gate_logits = (
                self.scene_replacement_gate_head(scene_gate_context)
                .squeeze(-1)
            )
            admission_logits = scene_replacement_gate_logits[:, None].expand(
                -1, num_expansion
            )
            centered_selector = candidate_selector_logits - (
                candidate_selector_logits.mean(dim=-1, keepdim=True)
            )
            base_floor = base_logits.amin(dim=-1, keepdim=True)
            selection_logits = (
                base_floor
                + self.hierarchical_selection_offset
                + self.hierarchical_selection_score_scale
                * torch.sigmoid(centered_selector)
            )
            replacement_utility = candidate_selector_logits
        elif self.use_set_marginal_gain_target:
            # A positive value means that this trajectory is predicted to
            # improve the complete protected set after replacing its lowest
            # confidence member. The same scalar drives admission and
            # ordering, matching the set-level training target below.
            base_floor = base_logits.amin(dim=-1, keepdim=True)
            admission_logits = legacy_admission_logits
            selection_logits = (
                base_floor
                + self.set_gain_score_scale
                * torch.tanh(admission_logits)
                + 0.0 * legacy_selection_logits
            )
        elif self.competitive_replacement_scoring:
            base_std = base_logits.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(0.25)
            base_floor = base_logits.amin(dim=-1, keepdim=True)
            if proposal_prior is None:
                proposal_prior = donor_score.new_zeros(donor_score.shape)
            utility_context = torch.cat(
                [
                    summary,
                    donor_score[..., None],
                    ((donor_logits - base_floor) / base_std)[..., None],
                    proposal_prior[..., None],
                ],
                dim=-1,
            )
            replacement_utility = self.replacement_utility_head(
                utility_context
            ).squeeze(-1)
            admission_logits = (
                replacement_utility
                / self.competitive_admission_temperature
            )
            selection_logits = (
                base_floor
                + self.competitive_utility_scale
                * torch.tanh(replacement_utility)
                # Keep legacy parameters in the DDP graph while the unified
                # utility head owns the actual deployment decision.
                + 0.0
                * (legacy_selection_logits + legacy_admission_logits)
            )
        else:
            admission_logits = legacy_admission_logits
            selection_logits = legacy_selection_logits
        calibrated_base_logits = base_logits
        unified_confidence = None
        if self.use_unified_deployed_confidence:
            unified_confidence = self._unified_confidence_logits(
                base_trajs=base_trajs,
                base_hidden=base_hidden,
                base_logits=base_logits,
                expansion_summary=summary,
                expansion_trajectory_token=trajectory_token,
                candidate_selector_logits=candidate_selector_logits,
            )
            calibrated_base_logits = unified_confidence["base_logits"]
            expansion_logits = unified_confidence["expansion_logits"]
            score_delta = unified_confidence["residual"][
                :, base_logits.shape[1] :
            ]
            confidence_gate = unified_confidence["gate"]
        elif self.use_decoupled_confidence:
            base_std = base_logits.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(0.25)
            base_floor = base_logits.amin(dim=-1, keepdim=True)
            score_context = torch.cat(
                [
                    summary,
                    donor_score[..., None],
                    ((donor_logits - base_floor) / base_std)[..., None],
                    torch.sigmoid(admission_logits)[..., None],
                    goal_gate[..., None],
                ],
                dim=-1,
            )
            score_delta = self.max_cross_score_delta * torch.tanh(
                self.cross_branch_score_head(score_context).squeeze(-1)
            )
            confidence_gate = torch.sigmoid(
                self.cross_branch_score_gate_logit
            )
            if self.cross_score_residual_from_selection:
                # Membership remains controlled by the frozen stage-one
                # selector.  The confidence branch is a strict zero-init
                # residual, so enabling it cannot change the starting model.
                expansion_logits = selection_logits + confidence_gate * (
                    self.cross_branch_score_bias + score_delta
                )
            else:
                calibrated_logits = (
                    donor_logits.detach()
                    + self.cross_branch_score_bias
                    + score_delta
                )
                expansion_logits = selection_logits + confidence_gate * (
                    calibrated_logits - selection_logits
                )
        else:
            score_delta = selection_score_delta
            confidence_gate = selection_logits.new_zeros(())
            expansion_logits = selection_logits
        return {
            "base_joint_logits": calibrated_base_logits,
            "joint_logits": expansion_logits,
            "selection_logits": selection_logits,
            "joint_trajs": expansion_trajs,
            "waypoint_delta": waypoint_delta,
            "dense_knot_delta": (
                dense_knot_delta
                if dense_knot_delta is not None
                else waypoint_delta.new_zeros(
                    batch_size, num_expansion, 2, 0, 2
                )
            ),
            "dense_temporal_delta": dense_temporal_delta,
            "score_delta": score_delta,
            "horizon_logits": horizon_logits,
            "admission_logits": admission_logits,
            "candidate_selector_logits": candidate_selector_logits,
            "scene_replacement_gate_logits": (
                scene_replacement_gate_logits
            ),
            "replacement_utility": replacement_utility,
            "confidence_gate": confidence_gate,
            "unified_confidence_residual": (
                unified_confidence["residual"]
                if unified_confidence is not None
                else expansion_logits.new_zeros(
                    batch_size,
                    base_logits.shape[1] + num_expansion,
                )
            ),
            "unified_confidence_temperature": (
                unified_confidence["temperature"]
                if unified_confidence is not None
                else expansion_logits.new_ones(batch_size)
            ),
            "goal_gate": goal_gate,
            "horizon_token": horizon_token,
            "trajectory_token": (
                trajectory_token
                if trajectory_token is not None
                else horizon_token.new_zeros(
                    batch_size, num_expansion, 0, self.hidden_dim
                )
            ),
        }


class WorldConditionedJointCandidateBank(nn.Module):
    """Build a joint intent bank before the expensive trajectory decoder.

    The trusted six-mode predictor remains the initialization, while two
    additional marginal intentions per target expose cross-agent combinations
    that the diagonal six-mode coupling cannot represent. Six output slots
    attend to all 8x8 joint pairs and progressively take over the initialization
    through learned gates.
    """

    def __init__(
        self,
        cfg,
        query_dim,
        hidden_dim,
        num_output_modes,
    ):
        super().__init__()
        self.model_cfg = cfg
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_output_modes = int(num_output_modes)
        self.num_marginal_candidates = int(
            cfg.get("NUM_MARGINAL_CANDIDATES", 8)
        )
        self.num_heads = int(cfg.get("NUM_HEADS", 8))
        self.dropout = float(cfg.get("DROPOUT", 0.1))
        self.assignment_temperature = float(
            cfg.get("ASSIGNMENT_TEMPERATURE", 0.7)
        )
        self.protected_pair_bias = float(
            cfg.get("PROTECTED_PAIR_BIAS", 6.0)
        )
        self.marginal_diversity_weight = float(
            cfg.get("MARGINAL_DIVERSITY_WEIGHT", 0.5)
        )
        self.marginal_diversity_scale = float(
            cfg.get("MARGINAL_DIVERSITY_SCALE", 10.0)
        )
        transport_cfg = cfg.get("SET_SLOT_TRANSPORT", {})
        self.use_set_slot_transport = bool(
            transport_cfg.get("ENABLED", False)
        )
        self.transport_gain_margin = float(
            transport_cfg.get("GAIN_MARGIN", 0.10)
        )
        self.transport_preserve_weight = float(
            transport_cfg.get("PRESERVE_WEIGHT", 0.20)
        )
        self.transport_assignment_margin = float(
            transport_cfg.get("ASSIGNMENT_MARGIN", 1.0)
        )
        self.transport_gate_bias_offset = float(
            transport_cfg.get("GATE_BIAS_OFFSET", 0.0)
        )
        self.use_global_unique_assignment = bool(
            transport_cfg.get("GLOBAL_UNIQUE_ASSIGNMENT", False)
        )
        self.force_identity_output = bool(
            transport_cfg.get("IDENTITY_WARMUP", False)
        )
        self.eps = 1e-6

        if self.num_marginal_candidates < self.num_output_modes:
            raise ValueError(
                "Candidate bank must retain at least one marginal candidate "
                "for every output mode"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "Candidate-bank hidden dimension must be divisible by heads"
            )
        if self.assignment_temperature <= 0.0:
            raise ValueError("ASSIGNMENT_TEMPERATURE must be positive")

        anchor_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anchor_encoder = nn.TransformerEncoder(
            anchor_layer,
            num_layers=int(cfg.get("NUM_ANCHOR_LAYERS", 1)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.agent_role_embedding = nn.Parameter(
            torch.empty(2, self.hidden_dim)
        )
        nn.init.normal_(self.agent_role_embedding, std=0.02)
        self.marginal_score_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.dropout,
        )
        # Start from geometry-only diverse extras and let the supervised
        # marginal objective learn scene-conditioned admission.
        nn.init.zeros_(self.marginal_score_head[-1].weight)
        nn.init.zeros_(self.marginal_score_head[-1].bias)

        self.physical_proj = _build_mlp(
            9,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
        )
        self.pair_fusion = _build_mlp(
            self.hidden_dim * 4,
            self.hidden_dim * 2,
            self.hidden_dim,
            self.dropout,
        )
        self.world_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.world_norm = nn.LayerNorm(self.hidden_dim)
        pair_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * int(
                cfg.get("FFN_MULTIPLIER", 3)
            ),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(
            pair_layer,
            num_layers=int(cfg.get("NUM_PAIR_LAYERS", 2)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.pair_score_head = _build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.dropout,
        )
        nn.init.zeros_(self.pair_score_head[-1].weight)
        nn.init.zeros_(self.pair_score_head[-1].bias)

        self.output_role_embedding = nn.Parameter(
            torch.empty(self.num_output_modes, self.hidden_dim)
        )
        nn.init.normal_(self.output_role_embedding, std=0.02)
        self.bank_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.bank_norm = nn.LayerNorm(self.hidden_dim)
        self.slot_query_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.pair_key_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.slot_transport_gate_head = None
        if self.use_set_slot_transport:
            self.slot_transport_gate_head = _build_mlp(
                self.hidden_dim * 3,
                self.hidden_dim,
                1,
                self.dropout,
            )
            nn.init.zeros_(self.slot_transport_gate_head[-1].weight)
            nn.init.zeros_(self.slot_transport_gate_head[-1].bias)
        self.joint_update = _build_mlp(
            self.hidden_dim * 3,
            self.hidden_dim * 2,
            self.hidden_dim,
            self.dropout,
        )
        self.query_content_update = _build_mlp(
            self.hidden_dim * 2,
            self.hidden_dim * 2,
            self.query_dim * 2,
            self.dropout,
        )
        nn.init.zeros_(self.joint_update[-1].weight)
        nn.init.zeros_(self.joint_update[-1].bias)
        nn.init.zeros_(self.query_content_update[-1].weight)
        nn.init.zeros_(self.query_content_update[-1].bias)

        self.anchor_gate_logit = nn.Parameter(
            torch.tensor(float(cfg.get("ANCHOR_GATE_BIAS_INIT", -4.0)))
        )
        self.context_gate_logit = nn.Parameter(
            torch.tensor(float(cfg.get("CONTEXT_GATE_BIAS_INIT", -3.0)))
        )

        pair_agent0 = []
        pair_agent1 = []
        protected = min(
            self.num_output_modes, self.num_marginal_candidates
        )
        protected_pairs = {(idx, idx) for idx in range(protected)}
        for idx in range(protected):
            pair_agent0.append(idx)
            pair_agent1.append(idx)
        for idx0 in range(self.num_marginal_candidates):
            for idx1 in range(self.num_marginal_candidates):
                if (idx0, idx1) in protected_pairs:
                    continue
                pair_agent0.append(idx0)
                pair_agent1.append(idx1)
        self.register_buffer(
            "pair_agent0_index",
            torch.tensor(pair_agent0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "pair_agent1_index",
            torch.tensor(pair_agent1, dtype=torch.long),
            persistent=False,
        )

    @property
    def num_joint_candidates(self):
        return self.num_marginal_candidates ** 2

    @staticmethod
    def _gather_agent_candidates(tensor, indices):
        view_shape = list(indices.shape) + [1] * (tensor.dim() - 3)
        expand_shape = list(indices.shape) + list(tensor.shape[3:])
        return tensor.gather(
            2, indices.view(*view_shape).expand(*expand_shape)
        )

    @staticmethod
    def _gather_modes(tensor, indices):
        view_shape = list(indices.shape) + [1] * (tensor.dim() - 2)
        expand_shape = list(indices.shape) + list(tensor.shape[2:])
        return tensor.gather(
            1, indices.view(*view_shape).expand(*expand_shape)
        )

    @staticmethod
    def _greedy_unique_pair_indices(logits):
        """Give every output slot a different joint pair."""
        batch_size, num_slots, num_pairs = logits.shape
        if num_slots > num_pairs:
            raise ValueError("More output slots than joint candidates")
        used = torch.zeros(
            batch_size,
            num_pairs,
            dtype=torch.bool,
            device=logits.device,
        )
        selected = []
        scores = logits.detach()
        for slot_idx in range(num_slots):
            available = scores[:, slot_idx].masked_fill(used, -torch.inf)
            pair_idx = available.argmax(dim=-1)
            selected.append(pair_idx)
            used.scatter_(1, pair_idx[:, None], True)
        return torch.stack(selected, dim=1)

    @staticmethod
    def _global_greedy_unique_pair_indices(logits):
        """Greedily match the globally strongest remaining slot-pair edge."""
        batch_size, num_slots, num_pairs = logits.shape
        if num_slots > num_pairs:
            raise ValueError("More output slots than joint candidates")
        assigned_slots = torch.zeros(
            batch_size,
            num_slots,
            dtype=torch.bool,
            device=logits.device,
        )
        used_pairs = torch.zeros(
            batch_size,
            num_pairs,
            dtype=torch.bool,
            device=logits.device,
        )
        selected = torch.full(
            (batch_size, num_slots),
            -1,
            dtype=torch.long,
            device=logits.device,
        )
        scores = logits.detach()
        for _ in range(num_slots):
            unavailable = (
                assigned_slots[:, :, None]
                | used_pairs[:, None, :]
            )
            flat_index = scores.masked_fill(
                unavailable, -torch.inf
            ).flatten(1).argmax(dim=-1)
            slot_index = torch.div(
                flat_index, num_pairs, rounding_mode="floor"
            )
            pair_index = flat_index.remainder(num_pairs)
            selected.scatter_(1, slot_index[:, None], pair_index[:, None])
            assigned_slots.scatter_(1, slot_index[:, None], True)
            used_pairs.scatter_(1, pair_index[:, None], True)
        return selected

    def _select_marginal_candidates(
        self,
        marginal_logits,
        anchor_points,
        protected_indices,
    ):
        """Keep trusted anchors and add geometrically distinct candidates."""
        batch_size, num_agents, num_anchors = marginal_logits.shape
        if self.num_marginal_candidates > num_anchors:
            raise ValueError(
                "Candidate bank asks for more anchors than are available"
            )
        num_protected = min(
            protected_indices.shape[-1],
            self.num_marginal_candidates,
        )
        selected = [
            protected_indices[..., idx] for idx in range(num_protected)
        ]
        used = torch.zeros(
            batch_size,
            num_agents,
            num_anchors,
            dtype=torch.bool,
            device=marginal_logits.device,
        )
        if selected:
            used.scatter_(
                2, torch.stack(selected, dim=-1), True
            )

        while len(selected) < self.num_marginal_candidates:
            selected_idx = torch.stack(selected, dim=-1)
            selected_points = self._gather_agent_candidates(
                anchor_points, selected_idx
            )
            distance = torch.linalg.vector_norm(
                anchor_points[:, :, :, None]
                - selected_points[:, :, None],
                dim=-1,
            ).min(dim=-1)[0]
            diversity_bonus = self.marginal_diversity_weight * torch.tanh(
                distance / max(self.marginal_diversity_scale, self.eps)
            )
            score = marginal_logits.detach() + diversity_bonus
            next_idx = score.masked_fill(used, -torch.inf).argmax(dim=-1)
            selected.append(next_idx)
            used.scatter_(2, next_idx[..., None], True)
        return torch.stack(selected, dim=-1)

    @staticmethod
    def _agent1_to_anchor0_points(agent1_points, pair_center_world):
        batch_size, num_pairs, _ = agent1_points.shape
        agent0_world = pair_center_world[:, 0]
        agent1_world = pair_center_world[:, 1]
        world_xy = common_utils.rotate_points_along_z(
            agent1_points,
            agent1_world[:, 6],
        )
        world_xy = world_xy + agent1_world[:, None, 0:2]
        return common_utils.rotate_points_along_z(
            world_xy - agent0_world[:, None, 0:2],
            -agent0_world[:, 6],
        ).reshape(batch_size, num_pairs, 2)

    def _build_pair_bank(
        self,
        selected_hidden,
        selected_query,
        selected_points,
        selected_logits,
        scene_token,
        memory,
        pair_state,
        pair_center_world,
    ):
        idx0 = self.pair_agent0_index.to(selected_hidden.device)
        idx1 = self.pair_agent1_index.to(selected_hidden.device)
        hidden0 = selected_hidden[:, 0].index_select(1, idx0)
        hidden1 = selected_hidden[:, 1].index_select(1, idx1)
        query0 = selected_query[:, 0].index_select(1, idx0)
        query1 = selected_query[:, 1].index_select(1, idx1)
        point0 = selected_points[:, 0].index_select(1, idx0)
        point1 = selected_points[:, 1].index_select(1, idx1)
        logit0 = selected_logits[:, 0].index_select(1, idx0)
        logit1 = selected_logits[:, 1].index_select(1, idx1)

        point1_canonical = self._agent1_to_anchor0_points(
            point1, pair_center_world
        )
        relative = point1_canonical - point0
        distance = torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        )
        speed = torch.linalg.vector_norm(
            pair_state[..., 2:4], dim=-1
        ) / 20.0
        speed = speed[:, None].expand(-1, point0.shape[1], -1)
        physical = torch.cat(
            [
                point0 / 50.0,
                point1_canonical / 50.0,
                relative / 50.0,
                distance / 20.0,
                speed,
            ],
            dim=-1,
        )
        physical_hidden = self.physical_proj(physical)
        scene = scene_token[:, None].expand(-1, point0.shape[1], -1)
        pair_hidden = self.pair_fusion(
            torch.cat(
                [hidden0, hidden1, physical_hidden, scene], dim=-1
            )
        )
        world_hidden, _ = self.world_cross_attention(
            pair_hidden,
            memory,
            memory,
            need_weights=False,
        )
        pair_hidden = self.world_norm(pair_hidden + world_hidden)
        pair_hidden = self.pair_encoder(pair_hidden)

        marginal_pair_logits = logit0 + logit1
        marginal_pair_logits = (
            marginal_pair_logits
            - marginal_pair_logits.mean(dim=-1, keepdim=True)
        ) / marginal_pair_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-3)
        pair_logits = (
            marginal_pair_logits
            + self.pair_score_head(pair_hidden).squeeze(-1)
        )
        pair_query = torch.stack([query0, query1], dim=2)
        pair_points_local = torch.stack([point0, point1], dim=2)
        pair_points_canonical = torch.stack(
            [point0, point1_canonical], dim=2
        )
        return (
            pair_hidden,
            pair_logits,
            pair_query,
            pair_points_local,
            pair_points_canonical,
        )

    def forward(
        self,
        anchor_hidden,
        anchor_query,
        anchor_points,
        base_query,
        base_points,
        base_query_content,
        base_joint_token,
        base_intent_assignment,
        state,
    ):
        batch_size, num_agents, num_anchors, _ = anchor_hidden.shape
        if num_agents != 2:
            raise ValueError("Joint candidate bank requires exactly 2 targets")

        scene = state["scene_token"]
        encoded_anchor = (
            anchor_hidden
            + scene[:, None, None]
            + self.agent_role_embedding[None, :, None]
        )
        encoded_anchor = self.anchor_encoder(
            encoded_anchor.reshape(
                batch_size * num_agents,
                num_anchors,
                self.hidden_dim,
            )
        ).reshape(
            batch_size, num_agents, num_anchors, self.hidden_dim
        )
        marginal_logits = self.marginal_score_head(
            encoded_anchor
        ).squeeze(-1)

        protected_indices = base_intent_assignment.argmax(
            dim=-1
        ).permute(0, 2, 1)
        selected_indices = self._select_marginal_candidates(
            marginal_logits,
            anchor_points,
            protected_indices,
        )
        selected_hidden = self._gather_agent_candidates(
            encoded_anchor, selected_indices
        )
        selected_query = self._gather_agent_candidates(
            anchor_query, selected_indices
        )
        selected_points = self._gather_agent_candidates(
            anchor_points, selected_indices
        )
        selected_logits = self._gather_agent_candidates(
            marginal_logits, selected_indices
        )

        (
            pair_hidden,
            pair_logits,
            pair_query,
            pair_points_local,
            pair_points_canonical,
        ) = self._build_pair_bank(
            selected_hidden=selected_hidden,
            selected_query=selected_query,
            selected_points=selected_points,
            selected_logits=selected_logits,
            scene_token=scene,
            memory=state["memory"],
            pair_state=state["pair_state"],
            pair_center_world=state["pair_center_world"],
        )

        slot = (
            base_joint_token
            + self.output_role_embedding[None]
            + scene[:, None]
        )
        bank_context, _ = self.bank_cross_attention(
            slot,
            pair_hidden,
            pair_hidden,
            need_weights=False,
        )
        slot_hidden = self.bank_norm(slot + bank_context)
        slot_logits = torch.einsum(
            "bqh,bph->bqp",
            self.slot_query_proj(slot_hidden),
            self.pair_key_proj(pair_hidden),
        ) / math.sqrt(self.hidden_dim)
        slot_logits = slot_logits + pair_logits[:, None]
        protected_bias = slot_logits.new_zeros(slot_logits.shape)
        diagonal = min(
            self.num_output_modes,
            slot_logits.shape[1],
            slot_logits.shape[2],
        )
        diagonal_idx = torch.arange(
            diagonal, device=slot_logits.device
        )
        protected_bias[:, diagonal_idx, diagonal_idx] = (
            self.protected_pair_bias
        )
        slot_logits = slot_logits + protected_bias
        soft_assignment = torch.softmax(
            slot_logits / self.assignment_temperature,
            dim=-1,
        )
        if self.use_global_unique_assignment:
            hard_pair_indices = self._global_greedy_unique_pair_indices(
                slot_logits
            )
        else:
            hard_pair_indices = self._greedy_unique_pair_indices(slot_logits)
        hard_assignment = F.one_hot(
            hard_pair_indices,
            num_classes=pair_hidden.shape[1],
        ).type_as(soft_assignment)
        pair_assignment = (
            hard_assignment
            + soft_assignment
            - soft_assignment.detach()
        )
        pair_log_probability = torch.log_softmax(pair_logits, dim=-1)
        slot_pair_log_prior = torch.einsum(
            "bqp,bp->bq", pair_assignment, pair_log_probability
        )

        candidate_query = torch.einsum(
            "bqp,bpad->bqad", pair_assignment, pair_query
        )
        candidate_points = torch.einsum(
            "bqp,bpad->bqad", pair_assignment, pair_points_local
        )
        candidate_pair_hidden = torch.einsum(
            "bqp,bph->bqh", pair_assignment, pair_hidden
        )
        if self.slot_transport_gate_head is not None:
            transport_gate_delta = self.slot_transport_gate_head(
                torch.cat(
                    [
                        slot_hidden,
                        candidate_pair_hidden,
                        candidate_pair_hidden - slot_hidden,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            slot_anchor_gate = torch.sigmoid(
                self.anchor_gate_logit
                + self.transport_gate_bias_offset
                + transport_gate_delta
            )
        else:
            transport_gate_delta = slot_logits.new_zeros(
                slot_logits.shape[:2]
            )
            slot_anchor_gate = torch.sigmoid(
                self.anchor_gate_logit
            ).expand(slot_logits.shape[:2])
        anchor_gate = slot_anchor_gate.mean()
        context_gate = torch.sigmoid(self.context_gate_logit)
        adapted_query = base_query + slot_anchor_gate[..., None, None] * (
            candidate_query - base_query
        )
        adapted_points = base_points + slot_anchor_gate[..., None, None] * (
            candidate_points - base_points
        )

        content_delta = self.query_content_update(
            torch.cat([slot_hidden, bank_context], dim=-1)
        ).reshape(
            batch_size,
            self.num_output_modes,
            2,
            self.query_dim,
        )
        adapted_query_content = (
            base_query_content + context_gate * content_delta
        )
        joint_delta = self.joint_update(
            torch.cat(
                [base_joint_token, slot_hidden, bank_context],
                dim=-1,
            )
        )
        adapted_joint_token = (
            base_joint_token + context_gate * joint_delta
        )

        pair_anchor_indices = torch.stack(
            [
                selected_indices[:, 0].index_select(
                    1, self.pair_agent0_index
                ),
                selected_indices[:, 1].index_select(
                    1, self.pair_agent1_index
                ),
            ],
            dim=2,
        )
        pair_anchor_assignment = F.one_hot(
            pair_anchor_indices,
            num_classes=num_anchors,
        ).type_as(pair_assignment)
        candidate_intent_assignment = torch.einsum(
            "bqp,bpak->bqak",
            pair_assignment,
            pair_anchor_assignment,
        )
        intent_gate = slot_anchor_gate[..., None, None]
        adapted_intent_assignment = (
            (1.0 - intent_gate) * base_intent_assignment
            + intent_gate * candidate_intent_assignment
        )
        if self.force_identity_output:
            # Stage one learns the enlarged bank and its assignment without
            # moving any deployed trajectory query.  This makes the warmup an
            # exact functional no-op for the trusted six-mode predictor.
            adapted_query = base_query
            adapted_points = base_points
            adapted_query_content = base_query_content
            adapted_joint_token = base_joint_token
            adapted_intent_assignment = base_intent_assignment
            slot_anchor_gate = slot_anchor_gate.new_zeros(
                slot_anchor_gate.shape
            )
            anchor_gate = slot_anchor_gate.mean()

        return {
            "adapted_query": adapted_query,
            "adapted_points": adapted_points,
            "adapted_query_content": adapted_query_content,
            "adapted_joint_token": adapted_joint_token,
            "adapted_intent_assignment": adapted_intent_assignment,
            "marginal_logits": marginal_logits,
            "anchor_points": anchor_points,
            "selected_indices": selected_indices,
            "pair_hidden": pair_hidden,
            "pair_logits": pair_logits,
            "pair_query": pair_query,
            "pair_points_local": pair_points_local,
            "pair_points_canonical": pair_points_canonical,
            "pair_anchor_assignment": pair_anchor_assignment,
            "slot_logits": slot_logits,
            "pair_assignment": pair_assignment,
            "hard_pair_indices": hard_pair_indices,
            "slot_pair_log_prior": slot_pair_log_prior,
            "anchor_gate": anchor_gate,
            "slot_anchor_gate": slot_anchor_gate,
            "transport_gate_delta": transport_gate_delta,
            "context_gate": context_gate,
        }

    def _build_set_slot_transport_target(
        self, pair_quality, pair_points_local
    ):
        """Replace one semantically nearest protected slot when it helps.

        The target is a valid partial permutation: unchanged slots retain
        their trusted diagonal pair, while a genuinely better non-diagonal
        pair is transported into the closest protected slot.  Unlike the
        former binary admission target, every scene supplies six assignment
        labels and every rescue scene supplies an explicit 64/144-way target.
        """
        batch_size, num_pairs = pair_quality.shape
        num_slots = min(self.num_output_modes, num_pairs)
        protected_quality = pair_quality[:, :num_slots]
        protected_best = protected_quality.min(dim=-1)[0]
        oracle_quality, oracle_pair = pair_quality.min(dim=-1)
        oracle_gain = (protected_best - oracle_quality).clamp_min(0.0)
        rescue_mask = (
            (oracle_pair >= num_slots)
            & (oracle_gain > self.transport_gain_margin)
        )

        protected_points = pair_points_local[:, :num_slots].reshape(
            batch_size, num_slots, -1
        )
        oracle_points = pair_points_local.gather(
            1,
            oracle_pair[:, None, None, None].expand(
                -1,
                1,
                pair_points_local.shape[2],
                pair_points_local.shape[3],
            ),
        ).reshape(batch_size, 1, -1)
        donor_slot = torch.cdist(
            protected_points.float(), oracle_points.float()
        ).squeeze(-1).argmin(dim=-1)

        target_assignment = torch.arange(
            num_slots, device=pair_quality.device
        )[None].expand(batch_size, -1).clone()
        batch_index = torch.arange(
            batch_size, device=pair_quality.device
        )
        target_assignment[batch_index, donor_slot] = torch.where(
            rescue_mask,
            oracle_pair,
            donor_slot,
        )
        return {
            "target_assignment": target_assignment,
            "oracle_pair": oracle_pair,
            "oracle_gain": oracle_gain,
            "rescue_mask": rescue_mask,
            "donor_slot": donor_slot,
        }

    def get_loss(self, ret, input_dict):
        anchor_points = ret["anchor_points"]
        batch_size = anchor_points.shape[0]
        gt = input_dict["center_gt_trajs"].to(
            anchor_points.device
        ).type_as(anchor_points)
        final_idx = input_dict["center_gt_final_valid_idx"].to(
            anchor_points.device
        ).long()
        flat_idx = torch.arange(
            gt.shape[0], device=gt.device
        )
        gt_goal = gt[flat_idx, final_idx, 0:2].reshape(
            batch_size, 2, 2
        )

        marginal_distance = torch.linalg.vector_norm(
            anchor_points - gt_goal[:, :, None],
            dim=-1,
        )
        marginal_target = marginal_distance.detach().argmin(dim=-1)
        loss_marginal = F.cross_entropy(
            ret["marginal_logits"].reshape(-1, marginal_distance.shape[-1]),
            marginal_target.reshape(-1),
        )

        selected_distance = self._gather_agent_candidates(
            marginal_distance, ret["selected_indices"]
        )
        pair_distance = torch.linalg.vector_norm(
            ret["pair_points_local"] - gt_goal[:, None],
            dim=-1,
        )
        pair_quality = pair_distance.mean(dim=-1)
        quality_temperature = float(
            self.model_cfg.get("QUALITY_TEMPERATURE", 1.5)
        )
        pair_target = torch.softmax(
            -pair_quality.detach() / max(quality_temperature, self.eps),
            dim=-1,
        )
        loss_pair = -(
            pair_target
            * torch.log_softmax(ret["pair_logits"], dim=-1)
        ).sum(dim=-1).mean()

        slot_probability = torch.softmax(
            ret["slot_logits"]
            / float(self.model_cfg.get("SLOT_LOSS_TEMPERATURE", 1.0)),
            dim=-1,
        )
        uncovered = (
            1.0 - slot_probability.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        coverage_probability = (1.0 - uncovered).clamp_min(self.eps)
        coverage_probability = coverage_probability / (
            coverage_probability.sum(dim=-1, keepdim=True)
            .clamp_min(self.eps)
        )
        loss_coverage = -(
            pair_target * coverage_probability.log()
        ).sum(dim=-1).mean()

        normalized_slot = F.normalize(slot_probability, p=2, dim=-1)
        similarity = torch.matmul(
            normalized_slot, normalized_slot.transpose(1, 2)
        )
        off_diagonal = ~torch.eye(
            similarity.shape[-1],
            device=similarity.device,
            dtype=torch.bool,
        )[None]
        loss_uniqueness = similarity.masked_select(
            off_diagonal.expand_as(similarity)
        ).mean()

        loss_slot_transport = pair_quality.new_zeros(())
        loss_transport_margin = pair_quality.new_zeros(())
        transport_metrics = {}
        if self.use_set_slot_transport:
            transport = self._build_set_slot_transport_target(
                pair_quality=pair_quality.detach(),
                pair_points_local=ret["pair_points_local"].detach(),
            )
            target_assignment = transport["target_assignment"]
            slot_cross_entropy = F.cross_entropy(
                ret["slot_logits"].reshape(-1, pair_quality.shape[-1]),
                target_assignment.reshape(-1),
                reduction="none",
            ).reshape_as(target_assignment)
            slot_weight = slot_cross_entropy.new_full(
                slot_cross_entropy.shape,
                self.transport_preserve_weight,
            )
            batch_index = torch.arange(
                batch_size, device=pair_quality.device
            )
            rescue_weight = transport["rescue_mask"].type_as(slot_weight)
            slot_weight[batch_index, transport["donor_slot"]] = torch.where(
                transport["rescue_mask"],
                torch.ones_like(rescue_weight),
                slot_weight[batch_index, transport["donor_slot"]],
            )
            loss_slot_transport = (
                slot_cross_entropy * slot_weight
            ).sum() / slot_weight.sum().clamp_min(self.eps)

            donor_logits = ret["slot_logits"][
                batch_index, transport["donor_slot"]
            ]
            target_logit = donor_logits.gather(
                1, transport["oracle_pair"][:, None]
            ).squeeze(1)
            protected_logit = donor_logits.gather(
                1, transport["donor_slot"][:, None]
            ).squeeze(1)
            margin_error = F.relu(
                self.transport_assignment_margin
                - target_logit
                + protected_logit
            )
            loss_transport_margin = (
                margin_error * rescue_weight
            ).sum() / rescue_weight.sum().clamp_min(1.0)

            target_probability = torch.softmax(
                donor_logits / self.assignment_temperature,
                dim=-1,
            ).gather(
                1, transport["oracle_pair"][:, None]
            ).squeeze(1)
            hard_donor_pair = ret["hard_pair_indices"][
                batch_index, transport["donor_slot"]
            ]
            transport_metrics = {
                "candidate_bank_transport_rescue_rate": (
                    rescue_weight.mean()
                ),
                "candidate_bank_transport_oracle_gain": (
                    transport["oracle_gain"].mean()
                ),
                "candidate_bank_transport_positive_gain": (
                    (
                        transport["oracle_gain"] * rescue_weight
                    ).sum()
                    / rescue_weight.sum().clamp_min(1.0)
                ),
                "candidate_bank_transport_target_probability": (
                    (target_probability * rescue_weight).sum()
                    / rescue_weight.sum().clamp_min(1.0)
                ),
                "candidate_bank_transport_target_recall": (
                    (
                        (hard_donor_pair == transport["oracle_pair"])
                        .type_as(rescue_weight)
                        * rescue_weight
                    ).sum()
                    / rescue_weight.sum().clamp_min(1.0)
                ),
                "candidate_bank_transport_assignment_accuracy": (
                    (
                        ret["hard_pair_indices"]
                        == target_assignment
                    ).float().mean()
                ),
            }

        total = (
            float(self.model_cfg.get("LOSS_WEIGHT_MARGINAL", 0.10))
            * loss_marginal
            + float(self.model_cfg.get("LOSS_WEIGHT_PAIR", 0.15))
            * loss_pair
            + float(self.model_cfg.get("LOSS_WEIGHT_COVERAGE", 0.10))
            * loss_coverage
            + float(self.model_cfg.get("LOSS_WEIGHT_UNIQUENESS", 0.02))
            * loss_uniqueness
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_SLOT_TRANSPORT", 0.0
                )
            )
            * loss_slot_transport
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_TRANSPORT_MARGIN", 0.0
                )
            )
            * loss_transport_margin
        )

        selected_pair_quality = self._gather_modes(
            pair_quality, ret["hard_pair_indices"]
        )
        marginal_hit = (
            ret["selected_indices"]
            == marginal_target[..., None]
        ).any(dim=-1).float().mean()
        pair_top1 = ret["pair_logits"].argmax(dim=-1)
        pair_top1_quality = pair_quality.gather(
            1, pair_top1[:, None]
        ).mean()
        metrics = {
            "loss_candidate_bank": total,
            "loss_candidate_bank_marginal": loss_marginal,
            "loss_candidate_bank_pair": loss_pair,
            "loss_candidate_bank_coverage": loss_coverage,
            "loss_candidate_bank_uniqueness": loss_uniqueness,
            "loss_candidate_bank_slot_transport": loss_slot_transport,
            "loss_candidate_bank_transport_margin": loss_transport_margin,
            "candidate_bank_marginal_recall": marginal_hit,
            "candidate_bank_marginal_oracle_fde": (
                selected_distance.min(dim=-1)[0].mean()
            ),
            "candidate_bank_pair_oracle_fde": pair_quality.min(
                dim=-1
            )[0].mean(),
            "candidate_bank_selected_oracle_fde": (
                selected_pair_quality.min(dim=-1)[0].mean()
            ),
            "candidate_bank_pair_top1_fde": pair_top1_quality,
            "candidate_bank_anchor_gate": ret["anchor_gate"],
            "candidate_bank_slot_anchor_gate_min": ret[
                "slot_anchor_gate"
            ].min(),
            "candidate_bank_slot_anchor_gate_max": ret[
                "slot_anchor_gate"
            ].max(),
            "candidate_bank_context_gate": ret["context_gate"],
        }
        metrics.update(transport_metrics)
        return total, metrics


class CandidateBankJointWorldDecoder(IntegratedJointWorldDecoder):
    """Integrated world decoder initialized from a full joint intent bank."""

    def __init__(
        self,
        cfg,
        query_dim,
        map_dim,
        num_future_frames,
        num_decoder_layers,
    ):
        super().__init__(
            cfg=cfg,
            query_dim=query_dim,
            map_dim=map_dim,
            num_future_frames=num_future_frames,
            num_decoder_layers=num_decoder_layers,
        )
        bank_cfg = cfg.get("CANDIDATE_BANK", {})
        if not bool(bank_cfg.get("ENABLED", True)):
            raise ValueError(
                "CandidateBankJointWorldDecoder requires "
                "CANDIDATE_BANK.ENABLED=True"
            )
        if not self.direct_sparse_modes:
            raise ValueError(
                "Candidate-bank decoder requires DIRECT_SPARSE_MODES=True"
            )
        if self.num_query_modes != self.num_output_modes:
            raise ValueError(
                "Candidate-bank decoder emits the official output count "
                "directly"
            )
        self.candidate_bank = WorldConditionedJointCandidateBank(
            cfg=bank_cfg,
            query_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
            num_output_modes=self.num_output_modes,
        )
        behavior_cfg = cfg.get("STRUCTURED_BEHAVIOR_WORLD", {})
        self.use_structured_behavior_world = bool(
            behavior_cfg.get("ENABLED", False)
        )
        self.structured_behavior_world_cfg = behavior_cfg
        self.structured_behavior_world_architecture = "disabled"
        if self.use_structured_behavior_world:
            raise ValueError(
                "STRUCTURED_BEHAVIOR_WORLD is not part of the public JFER release"
            )
        self.structured_behavior_world = None
        metric_cfg = cfg.get("METRIC_ALIGNED_SCORER", {})
        self.use_metric_aligned_scorer = bool(
            metric_cfg.get("ENABLED", False)
        )
        self.metric_aligned_scorer_cfg = metric_cfg
        self.metric_scorer = (
            WorldConditionedMetricScorer(
                cfg=metric_cfg,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                measurement_steps=cfg.get(
                    "MEASUREMENT_STEPS", [29, 49, 79]
                ),
                dt=self.dt,
            )
            if self.use_metric_aligned_scorer
            else None
        )
        expansion_cfg = cfg.get("PROTECTED_EXPANSION", {})
        self.use_candidate_expansion = bool(
            expansion_cfg.get("ENABLED", False)
        )
        if self.use_candidate_expansion and not self.expand_after_decoder:
            raise ValueError(
                "Candidate-bank protected expansion must run after the decoder"
            )
        self.expansion_novelty_weight = float(
            expansion_cfg.get("BANK_NOVELTY_WEIGHT", 1.0)
        )
        self.expansion_diversity_weight = float(
            expansion_cfg.get("BANK_DIVERSITY_WEIGHT", 0.5)
        )
        self.expansion_distance_scale = float(
            expansion_cfg.get("BANK_DISTANCE_SCALE", 5.0)
        )
        expansion_generator = str(
            expansion_cfg.get("GENERATOR_TYPE", "protected_endpoint")
        ).lower()
        if expansion_generator != "protected_endpoint":
            raise ValueError(
                "The public JFER release supports protected_endpoint expansion only"
            )
        if self.use_candidate_expansion:
            self.candidate_expansion = CandidateBankProtectedExpansion(
                cfg=expansion_cfg,
                hidden_dim=self.hidden_dim,
                num_expansion_modes=self.num_expansion_modes,
                measurement_steps=cfg.get(
                    "MEASUREMENT_STEPS", [29, 49, 79]
                ),
                num_future_frames=self.num_future_frames,
            )
        else:
            self.candidate_expansion = None
        residual_flow_cfg = cfg.get("CANDIDATE_RESIDUAL_FLOW", {})
        self.use_candidate_residual_flow = bool(
            residual_flow_cfg.get("ENABLED", False)
        )
        if self.use_candidate_residual_flow and not self.use_candidate_expansion:
            raise ValueError(
                "CANDIDATE_RESIDUAL_FLOW requires protected expansion"
            )
        self.candidate_residual_flow_cfg = residual_flow_cfg
        self.candidate_residual_flow = (
            WorldConditionedCandidateResidualFlow(
                cfg=residual_flow_cfg,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                measurement_steps=cfg.get(
                    "MEASUREMENT_STEPS", [29, 49, 79]
                ),
                num_future_frames=self.num_future_frames,
                dt=self.dt,
            )
            if self.use_candidate_residual_flow
            else None
        )
        cwrr_cfg = cfg.get("CWRR", {})
        self.use_cwrr = bool(cwrr_cfg.get("ENABLED", False))
        if self.use_cwrr:
            raise ValueError("CWRR is not part of the public JFER release")
        self.cwrr_cfg = cwrr_cfg
        self.cwrr = None
        self.use_improvement_gated_residual_flow = bool(
            self.candidate_residual_flow is not None
            and self.candidate_residual_flow.use_improvement_gated
        )
        b2_confidence_cfg = cfg.get("JFER_B2_CONFIDENCE", {})
        self.use_jfer_b2_confidence = bool(
            b2_confidence_cfg.get("ENABLED", False)
        )
        if self.use_jfer_b2_confidence:
            raise ValueError("JFER_B2_CONFIDENCE is not part of this release")
        self.jfer_b2_confidence_cfg = b2_confidence_cfg
        self.jfer_b2_confidence = None
        wcpr_cfg = cfg.get("WCPR", {})
        self.use_wcpr = bool(wcpr_cfg.get("ENABLED", False))
        if self.use_wcpr:
            raise ValueError("WCPR is not part of the public JFER release")
        self.wcpr_cfg = wcpr_cfg
        self.wcpr = None

        # Reciprocal augmentation is deliberately separate from the verified
        # protected expansion and residual-flow bank.  It consumes the six
        # trajectories that the legacy policy would actually deploy, then
        # proposes one confidence-preserving replacement or an explicit no-op.
        reciprocal_cfg = cfg.get("RECIPROCAL_AUGMENTATION", {})
        self.use_reciprocal_augmentation = bool(
            reciprocal_cfg.get("ENABLED", False)
        )
        self.reciprocal_augmentation_cfg = reciprocal_cfg
        self.reciprocal_candidate_expansion = None
        self.reciprocal_set_selector = None
        self.reciprocal_world_set_decoder = None
        self.reciprocal_set_selector_cfg = reciprocal_cfg.get(
            "RELATIVE_SELECTOR", {}
        )
        self.reciprocal_world_set_decoder_cfg = reciprocal_cfg.get(
            "JOINT_SET_DECODER", {}
        )
        self.use_reciprocal_world_set_decoder = bool(
            self.reciprocal_world_set_decoder_cfg.get("ENABLED", False)
        )
        if self.use_reciprocal_augmentation:
            raise ValueError(
                "RECIPROCAL_AUGMENTATION is not part of the public JFER release"
            )
        time_warp_cfg = cfg.get("INTERACTION_TIME_WARP_BANK", {})
        self.use_interaction_time_warp_bank = bool(
            time_warp_cfg.get("ENABLED", False)
        )
        if self.use_interaction_time_warp_bank:
            raise ValueError(
                "INTERACTION_TIME_WARP_BANK is not part of the public JFER release"
            )
        self.interaction_time_warp_cfg = time_warp_cfg
        self.interaction_time_warp_bank = None
        deployed_set_cfg = cfg.get("DEPLOYED_SET_SCORER", {})
        self.use_deployed_set_scorer = bool(
            deployed_set_cfg.get("ENABLED", False)
        )
        if self.use_deployed_set_scorer and not self.use_candidate_expansion:
            raise ValueError(
                "DEPLOYED_SET_SCORER requires PROTECTED_EXPANSION.ENABLED=True"
            )
        self.deployed_set_scorer_cfg = deployed_set_cfg
        self.deployed_set_scorer = (
            DeployedSetUtilityScorer(
                cfg=deployed_set_cfg,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                measurement_steps=cfg.get(
                    "MEASUREMENT_STEPS", [29, 49, 79]
                ),
                dt=self.dt,
            )
            if self.use_deployed_set_scorer
            else None
        )
        deployed_refiner_cfg = cfg.get(
            "DEPLOYED_TRAJECTORY_REFINER", {}
        )
        self.use_deployed_trajectory_refiner = bool(
            deployed_refiner_cfg.get("ENABLED", False)
        )
        if (
            self.use_deployed_trajectory_refiner
            and not self.use_candidate_expansion
        ):
            raise ValueError(
                "DEPLOYED_TRAJECTORY_REFINER requires "
                "PROTECTED_EXPANSION.ENABLED=True"
            )
        self.deployed_trajectory_refiner_cfg = deployed_refiner_cfg
        self.deployed_trajectory_refiner = (
            ConfidencePreservingDeployedTrajectoryRefiner(
                cfg=deployed_refiner_cfg,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                measurement_steps=cfg.get(
                    "MEASUREMENT_STEPS", [29, 49, 79]
                ),
                num_future_frames=self.num_future_frames,
                dt=self.dt,
            )
            if self.use_deployed_trajectory_refiner
            else None
        )
        if (
            self.use_improvement_gated_residual_flow
            and self.use_deployed_trajectory_refiner
        ):
            raise ValueError(
                "Improvement-gated JFER and DEPLOYED_TRAJECTORY_REFINER "
                "cannot both own the final-six geometry"
            )
        transport_cfg = cfg.get("SET_TO_SLOT_TRANSPORT", {})
        self.use_set_to_slot_transport = bool(
            transport_cfg.get("ENABLED", False)
        )
        if self.use_set_to_slot_transport and not self.use_candidate_expansion:
            raise ValueError(
                "SET_TO_SLOT_TRANSPORT requires protected expansion"
            )
        if (
            self.use_set_to_slot_transport
            and self.use_deployed_trajectory_refiner
        ):
            raise ValueError(
                "Set-to-slot transport must precede any future all-bank "
                "trajectory refiner"
            )
        self.set_to_slot_transport_cfg = transport_cfg
        if self.use_set_to_slot_transport:
            raise ValueError(
                "SET_TO_SLOT_TRANSPORT is not part of the public JFER release"
            )
        self.set_to_slot_transport = None
        sequential_cfg = cfg.get("SEQUENTIAL_MODE_DECODER", {})
        self.use_sequential_mode_decoder = bool(
            sequential_cfg.get("ENABLED", False)
        )
        if self.use_sequential_mode_decoder and not self.use_candidate_expansion:
            raise ValueError(
                "SEQUENTIAL_MODE_DECODER requires protected expansion"
            )
        if self.use_sequential_mode_decoder and any(
            module is not None
            for module in (
                self.candidate_residual_flow,
                self.deployed_set_scorer,
                self.deployed_trajectory_refiner,
                self.set_to_slot_transport,
            )
        ):
            raise ValueError(
                "SEQUENTIAL_MODE_DECODER owns 12-to-6 selection and "
                "refinement; legacy residual/set modules must be disabled"
            )
        self.sequential_mode_decoder_cfg = sequential_cfg
        if self.use_sequential_mode_decoder:
            raise ValueError(
                "SEQUENTIAL_MODE_DECODER is not part of the public JFER release"
            )
        self.sequential_mode_decoder = None
        if self.use_reciprocal_augmentation:
            if self.candidate_residual_flow is None:
                raise ValueError(
                    "RECIPROCAL_AUGMENTATION requires the verified candidate "
                    "residual-flow bank"
                )
            conflicting = {
                "INTERACTION_TIME_WARP_BANK": self.interaction_time_warp_bank,
                "DEPLOYED_SET_SCORER": self.deployed_set_scorer,
                "DEPLOYED_TRAJECTORY_REFINER": (
                    self.deployed_trajectory_refiner
                ),
                "SET_TO_SLOT_TRANSPORT": self.set_to_slot_transport,
                "SEQUENTIAL_MODE_DECODER": self.sequential_mode_decoder,
                "STRUCTURED_BEHAVIOR_WORLD": self.structured_behavior_world,
            }
            active = [name for name, module in conflicting.items() if module]
            if active:
                raise ValueError(
                    "RECIPROCAL_AUGMENTATION owns the final deployment step; "
                    "disable conflicting modules: " + ", ".join(active)
                )

    @property
    def coupled_candidate_transport(self):
        return (
            self.use_set_to_slot_transport
            and self.train_stage == "candidate_bank_set_to_slot_joint"
        )

    def train_metric_scorer_modules(self):
        if self.metric_scorer is None:
            raise RuntimeError(
                "candidate_bank_ap_calibration requires "
                "METRIC_ALIGNED_SCORER.ENABLED=True"
            )
        self.training = False
        self.metric_scorer.train(True)

    def train_candidate_residual_flow_modules(self, score_only=False):
        if self.candidate_residual_flow is None:
            raise RuntimeError(
                "candidate residual-flow training requires "
                "CANDIDATE_RESIDUAL_FLOW.ENABLED=True"
            )
        self.training = False
        self.candidate_residual_flow.train(not score_only)
        if score_only:
            self.candidate_residual_flow.horizon_score_head.train(True)

    def train_improvement_gated_residual_flow_modules(self, phase):
        if not self.use_improvement_gated_residual_flow:
            raise RuntimeError(
                "candidate_bank_improvement_gated_flow requires "
                "CANDIDATE_RESIDUAL_FLOW.IMPROVEMENT_GATED.ENABLED=True"
            )
        self.training = False
        self.candidate_residual_flow.set_improvement_train_phase(phase)

    def train_cwrr_response_modules(self):
        if self.cwrr is None:
            raise RuntimeError(
                "CWRR response training requires CWRR.ENABLED=True"
            )
        self.training = False
        self.cwrr.training = True
        self.cwrr.response_decoder.train(True)
        self.cwrr.refiner.eval()

    def train_cwrr_refinement_modules(self):
        if self.cwrr is None or not self.cwrr.enable_refinement:
            raise RuntimeError(
                "CWRR refinement training requires ENABLE_REFINEMENT=True"
            )
        self.training = False
        self.cwrr.training = True
        self.cwrr.response_decoder.eval()
        self.cwrr.refiner.train(True)

    @staticmethod
    def _jfer_full_convergence_phase(input_dict):
        """Return the zero-based formal JFER phase for the current epoch."""
        epoch = int(input_dict.get("cur_epoch", 0))
        if epoch < 2:
            return 0  # Phase A: geometry stabilization.
        if epoch < 8:
            return 1  # Phase B: confidence and selection activation.
        if epoch < 24:
            return 2  # Phase C: candidate co-adaptation.
        return 3  # Phase D: joint low-LR convergence.

    def train_jfer_full_convergence_modules(self, phase):
        """Set train/eval mode without changing the formal parameter policy."""
        if self.candidate_residual_flow is None:
            raise RuntimeError(
                "JFER full convergence requires CANDIDATE_RESIDUAL_FLOW"
            )
        self.training = False
        self.candidate_residual_flow.train(True)
        if phase >= 1:
            if self.metric_scorer is not None:
                self.metric_scorer.train(True)
            if self.candidate_expansion is None:
                raise RuntimeError(
                    "JFER full convergence requires protected expansion"
                )
            # Keep expansion geometry deterministic in Phase B while allowing
            # only the existing deployed score/admission heads to use dropout.
            for module_name in (
                "score_head",
                "horizon_score_head",
                "admission_head",
                "cross_branch_score_head",
                "scene_replacement_gate_head",
                "replacement_utility_head",
            ):
                module = getattr(self.candidate_expansion, module_name, None)
                if module is not None:
                    module.train(True)
        if phase >= 2:
            self.candidate_bank.train(True)
            self.candidate_expansion.train(True)

    def train_structured_behavior_world_modules(self):
        if self.structured_behavior_world is None:
            raise RuntimeError(
                "structured behavior warmup requires "
                "STRUCTURED_BEHAVIOR_WORLD.ENABLED=True"
            )
        self.training = False
        if (
            getattr(
                self.structured_behavior_world,
                "use_unified_success_posterior",
                False,
            )
            and not getattr(
                self.structured_behavior_world,
                "posterior_train_bank",
                False,
            )
        ):
            self.structured_behavior_world.train_unified_posterior_modules()
        else:
            self.structured_behavior_world.train(True)

    def train_permutation_scorer_modules(self):
        if (
            self.metric_scorer is None
            or not self.metric_scorer.use_score_multiset_permutation
        ):
            raise RuntimeError(
                "candidate_bank_permutation_calibration requires "
                "METRIC_ALIGNED_SCORER.SCORE_MULTISET_PERMUTATION=True"
            )
        self.training = False
        self.metric_scorer.training = False
        for module in (
            self.metric_scorer.permutation_input_norm,
            self.metric_scorer.permutation_input_proj,
            self.metric_scorer.permutation_set_encoder,
            self.metric_scorer.permutation_type_heads,
        ):
            module.train(True)

    def train_candidate_expansion_modules(self):
        if self.candidate_expansion is None:
            raise RuntimeError(
                "candidate_bank_expansion requires "
                "PROTECTED_EXPANSION.ENABLED=True"
            )
        self.training = False
        self.candidate_expansion.train(True)

    def train_reciprocal_augmentation_modules(self):
        if (
            self.reciprocal_candidate_expansion is None
            or self.reciprocal_set_selector is None
        ):
            raise RuntimeError(
                "candidate_bank_reciprocal_relative requires "
                "RECIPROCAL_AUGMENTATION.ENABLED=True"
            )
        # The checkpointed JFER proposal/deployment path remains deterministic.
        # Only the causal proposal generator and its relative replacement
        # policy use train-mode dropout and receive optimizer updates.
        self.training = False
        self.candidate_expansion.eval()
        self.candidate_residual_flow.eval()
        self.reciprocal_candidate_expansion.train(True)
        self.reciprocal_set_selector.train(True)

    def train_reciprocal_generator_modules(self):
        if (
            self.reciprocal_candidate_expansion is None
        ):
            raise RuntimeError(
                "candidate_bank_reciprocal_generator requires "
                "RECIPROCAL_AUGMENTATION.ENABLED=True"
            )
        self.training = False
        self.candidate_expansion.eval()
        self.candidate_residual_flow.eval()
        self.reciprocal_candidate_expansion.train(True)
        if self.reciprocal_set_selector is not None:
            self.reciprocal_set_selector.eval()
        if self.reciprocal_world_set_decoder is not None:
            self.reciprocal_world_set_decoder.eval()

    def train_reciprocal_selector_modules(self):
        if (
            self.reciprocal_candidate_expansion is None
            or self.reciprocal_set_selector is None
        ):
            raise RuntimeError(
                "candidate_bank_reciprocal_selector requires "
                "RECIPROCAL_AUGMENTATION.ENABLED=True"
            )
        self.training = False
        self.candidate_expansion.eval()
        self.candidate_residual_flow.eval()
        self.reciprocal_candidate_expansion.eval()
        self.reciprocal_set_selector.train(True)

    def train_reciprocal_world_set_decoder_modules(self, joint=False):
        if (
            self.reciprocal_candidate_expansion is None
            or self.reciprocal_world_set_decoder is None
        ):
            raise RuntimeError(
                "reciprocal world set training requires "
                "RECIPROCAL_AUGMENTATION.JOINT_SET_DECODER.ENABLED=True"
            )
        self.training = False
        self.candidate_expansion.eval()
        self.candidate_residual_flow.eval()
        self.reciprocal_candidate_expansion.train(bool(joint))
        if self.reciprocal_set_selector is not None:
            self.reciprocal_set_selector.eval()
        self.reciprocal_world_set_decoder.train(True)

    def train_interaction_time_warp_modules(self):
        if self.interaction_time_warp_bank is None:
            raise RuntimeError(
                "candidate_bank_interaction_time_warp requires "
                "INTERACTION_TIME_WARP_BANK.ENABLED=True"
            )
        self.training = False
        self.interaction_time_warp_bank.train(True)

    def train_interaction_set_replacement_modules(self):
        module = self.interaction_time_warp_bank
        if module is None or module.replacement_selector is None:
            raise RuntimeError(
                "candidate_bank_interaction_set_replacement requires "
                "SET_MARGINAL_REPLACEMENT.ENABLED=True"
            )
        # Candidate geometry must be identical to evaluation. Only the new
        # action head uses train-mode dropout and receives gradients.
        self.training = False
        module.eval()
        module.replacement_selector.train(True)

    def train_candidate_expansion_calibration_modules(self):
        expansion = self.candidate_expansion
        if expansion is None or not expansion.use_decoupled_confidence:
            raise RuntimeError(
                "candidate_bank_expansion_calibration requires decoupled "
                "candidate confidence"
            )
        self.training = False
        expansion.training = False
        expansion.cross_branch_score_head.train(True)

    def train_deployed_set_scorer_modules(self):
        if self.deployed_set_scorer is None:
            raise RuntimeError(
                "candidate_bank_set_utility_calibration requires "
                "DEPLOYED_SET_SCORER.ENABLED=True"
            )
        self.training = False
        self.deployed_set_scorer.train(True)

    def train_deployed_trajectory_refiner_modules(self):
        if self.deployed_trajectory_refiner is None:
            raise RuntimeError(
                "candidate_bank_deployed_geometry_refinement requires "
                "DEPLOYED_TRAJECTORY_REFINER.ENABLED=True"
            )
        self.training = False
        self.deployed_trajectory_refiner.train(True)

    def train_set_to_slot_transport_modules(self):
        if self.set_to_slot_transport is None:
            raise RuntimeError(
                "candidate_bank_set_to_slot_retrieval requires "
                "SET_TO_SLOT_TRANSPORT.ENABLED=True"
            )
        self.training = False
        self.set_to_slot_transport.train(True)

    def train_sequential_mode_modules(self, refiner_only=False):
        module = self.sequential_mode_decoder
        if module is None:
            raise RuntimeError(
                "sequential-mode training requires "
                "SEQUENTIAL_MODE_DECODER.ENABLED=True"
            )
        self.training = False
        if refiner_only:
            module.training = False
            if module.refinement_world_branch_energy is not None:
                # Keep the expert bank and its trajectory/world encoder fixed
                # while calibrating branch utility.  Otherwise both the input
                # and the supervision target move on every optimizer step.
                children = [
                    module.refinement_world_branch_embedding,
                    module.refinement_world_branch_energy,
                ]
            else:
                children = [
                    module.trajectory_proj,
                    module.world_proj,
                    module.spatial_proj,
                    module.scene_proj,
                    module.object_proj,
                    module.map_proj,
                    module.context_position_proj,
                    module.context_source_embedding,
                    module.tube_attention,
                    module.tube_attention_norm,
                    module.temporal_encoder,
                    module.score_proj,
                    module.branch_embedding,
                    module.type_embedding,
                    module.set_encoder,
                    module.initial_state,
                    module.selection_gru,
                    module.refine_slot_proj,
                    module.refine_encoder,
                ]
                if module.refinement_expert_head is not None:
                    children.append(module.refinement_expert_head)
                    children.extend(
                        [
                            module.refinement_expert_delta_proj,
                            module.refinement_expert_embedding,
                            module.refinement_expert_score,
                        ]
                    )
                    if module.refinement_admission_score is not None:
                        children.append(module.refinement_admission_score)
                else:
                    children.extend([module.refine_head, module.refine_gate])
            for child in children:
                child.train(True)
        else:
            module.train(True)

    def train_final_permutation_modules(self):
        scorer = self.deployed_set_scorer
        if scorer is None or not scorer.use_score_multiset_permutation:
            raise RuntimeError(
                "candidate_bank_final_permutation_calibration requires "
                "DEPLOYED_SET_SCORER.SCORE_MULTISET_PERMUTATION=True"
            )
        self.training = False
        scorer.training = False
        for module in (
            scorer.world_norm,
            scorer.scene_norm,
            scorer.trajectory_proj,
            scorer.spatial_proj,
            scorer.score_proj,
            scorer.type_embedding,
            scorer.set_encoder,
            scorer.horizon_encoder,
            scorer.permutation_input_norm,
            scorer.permutation_set_encoder,
            scorer.permutation_type_heads,
            scorer.permutation_acceptance_head,
        ):
            module.train(True)

    @staticmethod
    def _pair_type_ids(input_dict, device, batch_size):
        return official_pair_type_ids(
            input_dict.get("pair_object_types"),
            batch_size,
            device,
        )

    @staticmethod
    def _gather_with_local_gradient(tensor):
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        local = tensor.contiguous()
        gathered = [torch.empty_like(local) for _ in range(
            dist.get_world_size()
        )]
        dist.all_gather(gathered, local.detach())
        gathered[dist.get_rank()] = local
        return torch.cat(gathered, dim=0)

    def prepare_worlds(
        self,
        center_feature,
        obj_feature,
        obj_mask,
        map_feature,
        map_mask,
        input_dict,
        obj_pos=None,
        map_pos=None,
    ):
        state = super().prepare_worlds(
            center_feature=center_feature,
            obj_feature=obj_feature,
            obj_mask=obj_mask,
            map_feature=map_feature,
            map_mask=map_mask,
            input_dict=input_dict,
            obj_pos=obj_pos,
            map_pos=map_pos,
        )
        if self.wcpr is not None:
            if obj_pos is None or map_pos is None:
                raise RuntimeError("WCPR requires object and map positions")
            batch_size = state["batch_size"]
            object_mask = self._reshape_pair(
                obj_mask, batch_size
            ).detach().clone()
            track_index = torch.as_tensor(
                input_dict["track_index_to_predict"],
                device=object_mask.device,
                dtype=torch.long,
            ).reshape(batch_size, 2)
            batch_index = torch.arange(
                batch_size, device=object_mask.device
            )[:, None].expand(batch_size, 2)
            role_index = torch.arange(
                2, device=object_mask.device
            )[None].expand(batch_size, 2)
            object_mask[batch_index, role_index, track_index] = False

            history = input_dict["obj_trajs"].to(
                center_feature.device, non_blocking=True
            )
            history_mask = input_dict["obj_trajs_mask"].to(
                center_feature.device, non_blocking=True
            ).bool()
            time_index = torch.arange(
                history.shape[2], device=history.device
            ).view(1, 1, -1)
            last_index = time_index.masked_fill(~history_mask, -1).amax(dim=2)
            safe_index = last_index.clamp_min(0)
            gather_index = safe_index[..., None, None].expand(-1, -1, 1, 2)
            object_velocity = history[..., 25:27].gather(
                2, gather_index
            ).squeeze(2)
            object_velocity = torch.where(
                (last_index >= 0)[..., None],
                object_velocity,
                torch.zeros_like(object_velocity),
            )
            state["wcpr_context"] = {
                "object_feature": self._reshape_pair(
                    obj_feature, batch_size
                ).detach(),
                "object_mask": object_mask,
                "object_position": self._reshape_pair(
                    obj_pos[..., 0:2], batch_size
                ).detach(),
                "object_velocity": self._reshape_pair(
                    object_velocity, batch_size
                ).detach(),
                "map_feature": self._reshape_pair(
                    map_feature, batch_size
                ).detach(),
                "map_mask": self._reshape_pair(
                    map_mask, batch_size
                ).detach(),
                "map_position": self._reshape_pair(
                    map_pos[..., 0:2], batch_size
                ).detach(),
            }
        if self.sequential_mode_decoder is not None:
            batch_size = state["batch_size"]
            state["sequential_mode_context"] = {
                "obj_feature": self._reshape_pair(
                    obj_feature, batch_size
                )[:, 0].detach(),
                "obj_mask": self._reshape_pair(
                    obj_mask, batch_size
                )[:, 0].detach(),
                "obj_pos": (
                    self._reshape_pair(obj_pos, batch_size)[:, 0].detach()
                    if obj_pos is not None
                    else None
                ),
                "map_feature": self._reshape_pair(
                    map_feature, batch_size
                )[:, 0].detach(),
                "map_mask": self._reshape_pair(
                    map_mask, batch_size
                )[:, 0].detach(),
                "map_pos": (
                    self._reshape_pair(map_pos, batch_size)[:, 0].detach()
                    if map_pos is not None
                    else None
                ),
            }
        state["pair_center_world"] = input_dict[
            "pair_center_objects_world"
        ].to(center_feature.device).type_as(center_feature)
        if self.cwrr is not None:
            if obj_pos is None:
                raise RuntimeError("CWRR requires object positions")
            state["cwrr_context"] = self.cwrr.build_context(
                obj_feature=obj_feature,
                obj_mask=obj_mask,
                obj_pos=obj_pos,
                input_dict=input_dict,
                batch_size=state["batch_size"],
            )
        if self.structured_behavior_world is not None:
            pair_type_ids = self._pair_type_ids(
                input_dict, center_feature.device, state["batch_size"]
            )
            behavior_prepare_kwargs = {
                "scene_token": state["scene_token"],
                "pair_state": state["pair_state"],
                "pair_type_ids": pair_type_ids,
                "input_dict": input_dict,
            }
            if (
                self.structured_behavior_world_architecture
                in {
                    "supervised_joint_hypothesis",
                    "coverage_ordered_joint_hypothesis",
                    "candidate_energy_transport",
                    "balanced_coverage_transport",
                }
            ):
                behavior_prepare_kwargs.update(
                    {
                        "center_feature": center_feature,
                        "obj_feature": obj_feature,
                        "obj_mask": obj_mask,
                        "obj_pos": obj_pos,
                        "map_feature": map_feature,
                        "map_mask": map_mask,
                        "map_pos": map_pos,
                    }
                )
            state["structured_behavior_world"] = (
                self.structured_behavior_world.prepare(
                    **behavior_prepare_kwargs
                )
            )
        return state

    def initialize_queries(self, intention_query, intention_points, state):
        (
            base_query,
            base_points,
            base_query_content,
            state,
        ) = super().initialize_queries(
            intention_query=intention_query,
            intention_points=intention_points,
            state=state,
        )
        batch_size = state["batch_size"]
        num_intents = intention_query.shape[0]
        anchor_query = intention_query.permute(1, 0, 2).reshape(
            batch_size, 2, num_intents, self.query_dim
        )
        anchor_points = intention_points.permute(1, 0, 2).reshape(
            batch_size, 2, num_intents, 2
        )
        anchor_hidden = self.agent_query_proj(anchor_query)
        base_query_batched = base_query.permute(1, 0, 2).reshape(
            batch_size, 2, self.num_output_modes, self.query_dim
        ).permute(0, 2, 1, 3)
        base_points_batched = base_points.permute(1, 0, 2).reshape(
            batch_size, 2, self.num_output_modes, 2
        ).permute(0, 2, 1, 3)
        base_content_batched = base_query_content.permute(
            1, 0, 2
        ).reshape(
            batch_size, 2, self.num_output_modes, self.query_dim
        ).permute(0, 2, 1, 3)

        bank_ret = self.candidate_bank(
            anchor_hidden=anchor_hidden,
            anchor_query=anchor_query,
            anchor_points=anchor_points,
            base_query=base_query_batched,
            base_points=base_points_batched,
            base_query_content=base_content_batched,
            base_joint_token=state["joint_token"],
            base_intent_assignment=state["intent_assignment"],
            state=state,
        )
        if self.structured_behavior_world is not None:
            bank_ret = self.structured_behavior_world.initialize_bank(
                bank_ret, state
            )
        adapted_query = bank_ret["adapted_query"].permute(
            0, 2, 1, 3
        ).reshape(
            batch_size * 2, self.num_output_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        adapted_points = bank_ret["adapted_points"].permute(
            0, 2, 1, 3
        ).reshape(
            batch_size * 2, self.num_output_modes, 2
        ).permute(1, 0, 2).contiguous()
        adapted_content = bank_ret["adapted_query_content"].permute(
            0, 2, 1, 3
        ).reshape(
            batch_size * 2, self.num_output_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()

        state["joint_token"] = bank_ret["adapted_joint_token"]
        state["intent_assignment"] = bank_ret[
            "adapted_intent_assignment"
        ]
        assignment, _, _ = self._assign_worlds(
            state["joint_token"], state
        )
        state["assignment"] = assignment
        state["candidate_bank"] = bank_ret
        state["candidate_bank_log_prior"] = bank_ret[
            "slot_pair_log_prior"
        ]
        return (
            adapted_query,
            adapted_points,
            adapted_content,
            state,
        )

    def couple_queries(self, layer_idx, query_content, state):
        query_content, state = super().couple_queries(
            layer_idx=layer_idx,
            query_content=query_content,
            state=state,
        )
        if self.structured_behavior_world is not None:
            query_content, state = (
                self.structured_behavior_world.couple_queries(
                    layer_idx=layer_idx,
                    query_content=query_content,
                    state=state,
                )
            )
        return query_content, state

    def _select_complementary_pair_indices(
        self, bank_ret, base_trajs, base_hidden, base_logits
    ):
        coupled = self.coupled_candidate_transport
        pair_points = bank_ret["pair_points_canonical"]
        pair_logits = bank_ret["pair_logits"]
        if not coupled:
            pair_points = pair_points.detach()
            pair_logits = pair_logits.detach()
        batch_size, num_pairs = pair_logits.shape
        base_endpoint = base_trajs[..., -1, :].reshape(
            batch_size, self.num_output_modes, -1
        )
        if not coupled:
            base_endpoint = base_endpoint.detach()
        pair_endpoint = pair_points.reshape(batch_size, num_pairs, -1)
        nearest_base = torch.cdist(
            pair_endpoint.float(), base_endpoint.float()
        ).amin(dim=-1).type_as(pair_logits)
        normalized_logit = pair_logits - pair_logits.mean(
            dim=-1, keepdim=True
        )
        normalized_logit = normalized_logit / pair_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        proposal = self.candidate_expansion.score_pair_bank(
            pair_hidden=(
                bank_ret["pair_hidden"]
                if coupled
                else bank_ret["pair_hidden"].detach()
            ),
            pair_endpoints=pair_points,
            pair_logits=pair_logits,
            base_hidden=(base_hidden if coupled else base_hidden.detach()),
            base_logits=(base_logits if coupled else base_logits.detach()),
            base_trajs=(base_trajs if coupled else base_trajs.detach()),
        )
        proposal["selection_logits"] = (
            self.candidate_expansion.pair_selector_prior_weight
            * normalized_logit
            + self.candidate_expansion.pair_selector_novelty_weight
            * self.expansion_novelty_weight
            * torch.tanh(
                nearest_base
                / max(self.expansion_distance_scale, self.eps)
            )
            + proposal["selection_delta"]
        )

        if self.candidate_expansion.full_bank_decoding:
            if num_pairs != self.num_expansion_modes:
                raise RuntimeError(
                    "FULL_BANK_DECODING requires NUM_EXPANSION_MODES to "
                    "match the complete joint intention bank"
                )
            all_pairs = torch.arange(
                num_pairs,
                device=pair_logits.device,
                dtype=torch.long,
            )[None].expand(batch_size, -1)
            return all_pairs, proposal

        used = torch.zeros(
            batch_size,
            num_pairs,
            device=pair_logits.device,
            dtype=torch.bool,
        )
        if not self.candidate_expansion.allow_base_pair_reuse:
            used.scatter_(1, bank_ret["hard_pair_indices"].detach(), True)
        selected = []
        for _ in range(self.num_expansion_modes):
            score = proposal["selection_logits"]
            if selected:
                selected_index = torch.stack(selected, dim=1)
                selected_endpoint = pair_endpoint.gather(
                    1,
                    selected_index[..., None].expand(
                        -1, -1, pair_endpoint.shape[-1]
                    ),
                )
                diversity = torch.cdist(
                    pair_endpoint.float(), selected_endpoint.float()
                ).amin(dim=-1).type_as(pair_logits)
                score = score + self.expansion_diversity_weight * torch.tanh(
                    diversity
                    / max(self.expansion_distance_scale, self.eps)
                )
            next_index = score.masked_fill(used, -torch.inf).argmax(dim=-1)
            selected.append(next_index)
            used.scatter_(1, next_index[:, None], True)
        return torch.stack(selected, dim=1), proposal

    def _straight_through_expansion_assignment(
        self, selection_logits, hard_indices
    ):
        """Keep deployment pairs discrete while exposing selector gradients."""
        num_pairs = selection_logits.shape[-1]
        temperature = max(
            float(
                self.set_to_slot_transport_cfg.get(
                    "PAIR_TRANSPORT_TEMPERATURE", 0.7
                )
            ),
            self.eps,
        )
        used = torch.zeros_like(selection_logits, dtype=torch.bool)
        assignments = []
        for slot_idx in range(hard_indices.shape[1]):
            masked_logits = selection_logits.masked_fill(
                used.clone(), -1e4
            )
            soft = torch.softmax(masked_logits / temperature, dim=-1)
            index = hard_indices[:, slot_idx]
            hard = F.one_hot(index, num_classes=num_pairs).type_as(soft)
            assignments.append(hard + soft - soft.detach())
            used.scatter_(1, index[:, None], True)
        return torch.stack(assignments, dim=1)

    def _append_interaction_time_warp_expansions(
        self, output, state, input_dict
    ):
        module = self.interaction_time_warp_bank
        if module is None:
            return output
        base_trajs = output["joint_trajs"][:, : self.num_output_modes]
        base_logits = output["joint_logits"][:, : self.num_output_modes]
        time_warp = module(
            trajectories=base_trajs,
            base_logits=base_logits,
            world_response=output["response"][:, : self.num_output_modes],
            joint_context=state["joint_token"][:, : self.num_output_modes],
            scene_context=state["scene_token"],
            pair_type_ids=self._pair_type_ids(
                input_dict,
                base_logits.device,
                base_logits.shape[0],
            ),
            scene_memory=state["scene_memory"],
            world_rollout=state["prior_rollout"],
            world_logits=state["world_logits"],
            world_sample_indices=self.sample_indices,
        )
        expanded = dict(output)
        expanded["protected_base_joint_logits"] = base_logits
        expanded["protected_base_joint_trajs"] = base_trajs
        expanded["joint_trajs"] = torch.cat(
            [base_trajs, time_warp["trajectories"]], dim=1
        )
        expanded["joint_logits"] = torch.cat(
            [base_logits, time_warp["joint_logits"]], dim=1
        )
        expanded["protected_expansion_selection_logits"] = torch.cat(
            [base_logits, time_warp["selection_logits"]], dim=1
        )
        expanded["protected_expansion_admission_logits"] = time_warp[
            "admission_logits"
        ]
        expanded["protected_expansion_world_hidden"] = torch.cat(
            [output["response"], time_warp["world_hidden"]], dim=1
        )
        expanded["interaction_time_warp_bank"] = time_warp
        return expanded

    def _append_candidate_expansions(self, output, state):
        bank_ret = state["candidate_bank"]
        reciprocal = bool(
            getattr(
                self.candidate_expansion,
                "is_reciprocal_conditional",
                False,
            )
        )
        coupled = self.coupled_candidate_transport
        if reciprocal and coupled:
            raise RuntimeError(
                "Reciprocal conditional generation owns its causal parent "
                "assignment and cannot use set-to-slot transport"
            )
        base_trajs = output["joint_trajs"]
        base_logits = output["joint_logits"]
        base_hidden = output["response"]
        if not coupled:
            base_trajs = base_trajs.detach()
            base_logits = base_logits.detach()
            base_hidden = base_hidden.detach()
        pair_assignment = None
        pair_proposal = None
        if reciprocal:
            # Keep both causal directions for each of the three strongest
            # parent modes.  Unlike the old residual-expert path, no GT-derived
            # branch is selected before the official six-mode deployment.
            donor_indices = self.candidate_expansion.select_donor_indices(
                base_logits
            )
            donor_hidden = WorldConditionedJointCandidateBank._gather_modes(
                base_hidden, donor_indices
            )
            donor_logits = base_logits.gather(1, donor_indices)
            donor_trajs = WorldConditionedJointCandidateBank._gather_modes(
                base_trajs, donor_indices
            )
            pair_indices = donor_indices
            pair_hidden = donor_hidden
            pair_endpoints = donor_trajs[..., -1, :]
            proposal_prior = donor_logits.new_zeros(donor_logits.shape)
            proposal_trajs = donor_trajs
        else:
            pair_indices, pair_proposal = (
                self._select_complementary_pair_indices(
                    bank_ret,
                    output["joint_trajs"],
                    output["response"],
                    output["joint_logits"],
                )
            )
            if coupled:
                pair_assignment = (
                    self._straight_through_expansion_assignment(
                        pair_proposal["selection_logits"], pair_indices
                    )
                )
                pair_hidden = torch.einsum(
                    "bep,bph->beh",
                    pair_assignment,
                    bank_ret["pair_hidden"],
                )
                pair_endpoints = torch.einsum(
                    "bep,bpad->bead",
                    pair_assignment,
                    bank_ret["pair_points_canonical"],
                )
            else:
                pair_hidden = (
                    WorldConditionedJointCandidateBank._gather_modes(
                        bank_ret["pair_hidden"], pair_indices
                    ).detach()
                )
                pair_endpoints = (
                    WorldConditionedJointCandidateBank._gather_modes(
                        bank_ret["pair_points_canonical"], pair_indices
                    ).detach()
                )

            donor_indices = pair_proposal["donor_indices"].gather(
                1, pair_indices
            )
            donor_hidden = base_hidden.gather(
                1,
                donor_indices[..., None].expand(
                    -1, -1, base_hidden.shape[-1]
                ),
            )
            donor_logits = base_logits.gather(1, donor_indices)
            normalized_pair_prior = bank_ret["pair_logits"]
            if not coupled:
                normalized_pair_prior = normalized_pair_prior.detach()
            normalized_pair_prior = (
                normalized_pair_prior
                - normalized_pair_prior.mean(dim=-1, keepdim=True)
            )
            normalized_pair_prior = normalized_pair_prior / (
                normalized_pair_prior.std(
                    dim=-1, keepdim=True, unbiased=False
                ).clamp_min(0.25)
            )
            proposal_prior = normalized_pair_prior.gather(1, pair_indices)
            donor_trajs = base_trajs.gather(
                1,
                donor_indices[..., None, None, None].expand(
                    -1,
                    -1,
                    base_trajs.shape[2],
                    base_trajs.shape[3],
                    base_trajs.shape[4],
                ),
            )
            proposal_trajs = None
            if self.candidate_expansion.use_pair_prototype_as_donor:
                if coupled:
                    proposal_trajs = torch.einsum(
                        "bep,bpath->beath",
                        pair_assignment,
                        pair_proposal["prototype_trajs"],
                    )
                else:
                    proposal_trajs = (
                        WorldConditionedJointCandidateBank._gather_modes(
                            pair_proposal["prototype_trajs"], pair_indices
                        )
                    )
        expansion = self.candidate_expansion(
            pair_hidden=pair_hidden,
            pair_endpoints=pair_endpoints,
            donor_hidden=donor_hidden,
            donor_logits=donor_logits,
            donor_trajs=donor_trajs,
            proposal_trajs=proposal_trajs,
            proposal_prior=proposal_prior,
            base_hidden=base_hidden,
            base_logits=base_logits,
            base_trajs=base_trajs,
        )
        expanded = dict(output)
        expanded["joint_logits"] = torch.cat(
            [expansion["base_joint_logits"], expansion["joint_logits"]], dim=1
        )
        expanded["joint_trajs"] = torch.cat(
            [output["joint_trajs"], expansion["joint_trajs"]], dim=1
        )
        expanded["protected_base_joint_logits"] = output["joint_logits"]
        expanded["protected_base_joint_trajs"] = output["joint_trajs"]
        expanded["protected_expansion_selection_logits"] = torch.cat(
            [output["joint_logits"], expansion["selection_logits"]], dim=1
        )
        expanded["protected_expansion_admission_logits"] = expansion[
            "admission_logits"
        ]
        expanded["protected_expansion_candidate_selector_logits"] = expansion[
            "candidate_selector_logits"
        ]
        expanded["protected_expansion_scene_gate_logits"] = expansion[
            "scene_replacement_gate_logits"
        ]
        expanded["protected_expansion_replacement_utility"] = expansion[
            "replacement_utility"
        ]
        expanded["candidate_expansion_horizon_logits"] = expansion[
            "horizon_logits"
        ]
        expanded["candidate_expansion_score_delta"] = expansion[
            "score_delta"
        ]
        expanded["candidate_expansion_waypoint_delta"] = expansion[
            "waypoint_delta"
        ]
        expanded["candidate_expansion_dense_knot_delta"] = expansion[
            "dense_knot_delta"
        ]
        expanded["candidate_expansion_dense_temporal_delta"] = expansion[
            "dense_temporal_delta"
        ]
        expanded["candidate_expansion_goal_gate"] = expansion["goal_gate"]
        expanded["candidate_expansion_confidence_gate"] = expansion[
            "confidence_gate"
        ]
        expanded["candidate_expansion_unified_confidence_residual"] = (
            expansion["unified_confidence_residual"]
        )
        expanded["candidate_expansion_unified_confidence_temperature"] = (
            expansion["unified_confidence_temperature"]
        )
        base_world_hidden = output["response"][:, :, None].expand(
            -1,
            -1,
            expansion["horizon_token"].shape[2],
            -1,
        )
        expansion_world_hidden = expansion["horizon_token"]
        if not coupled:
            base_world_hidden = base_world_hidden.detach()
            expansion_world_hidden = expansion_world_hidden.detach()
        expanded["protected_expansion_world_hidden"] = torch.cat(
            [base_world_hidden, expansion_world_hidden], dim=1
        )
        if pair_assignment is not None:
            expanded["candidate_expansion_pair_assignment"] = pair_assignment
        expanded["candidate_expansion_pair_indices"] = pair_indices
        expanded["candidate_expansion_donor_indices"] = donor_indices
        if reciprocal:
            expanded["candidate_expansion_influencer_indices"] = expansion[
                "influencer_indices"
            ]
            expanded["candidate_expansion_reactor_indices"] = expansion[
                "reactor_indices"
            ]
            expanded["candidate_expansion_interaction_probability"] = (
                expansion["interaction_probability"]
            )
        if pair_proposal is not None:
            factorized_donor_indices = pair_proposal.get(
                "factorized_donor_indices"
            )
            if factorized_donor_indices is not None:
                expanded["candidate_expansion_factorized_donor_indices"] = (
                    WorldConditionedJointCandidateBank._gather_modes(
                        factorized_donor_indices, pair_indices
                    )
                )
            expanded["candidate_pair_proposal_horizon_logits"] = (
                pair_proposal["horizon_logits"]
            )
            expanded["candidate_pair_proposal_rescue_logits"] = pair_proposal[
                "rescue_logits"
            ]
            expanded["candidate_pair_proposal_quality_logits"] = pair_proposal[
                "quality_logits"
            ]
            expanded["candidate_pair_proposal_selection_delta"] = (
                pair_proposal["selection_delta"]
            )
            expanded["candidate_pair_proposal_selection_logits"] = (
                pair_proposal["selection_logits"]
            )
            expanded["candidate_pair_proposal_gate"] = pair_proposal["gate"]
            expanded["candidate_pair_prototype_trajs"] = pair_proposal[
                "prototype_trajs"
            ]
            expanded["candidate_pair_proposal_temporal_delta"] = (
                pair_proposal["temporal_waypoint_delta"]
            )
            pair_used_mask = torch.zeros_like(
                bank_ret["pair_logits"], dtype=torch.bool
            )
            if not self.candidate_expansion.allow_base_pair_reuse:
                pair_used_mask.scatter_(
                    1, bank_ret["hard_pair_indices"].detach(), True
                )
            expanded["candidate_pair_base_used_mask"] = pair_used_mask
        return expanded

    def _apply_candidate_residual_flow(self, output, state, input_dict):
        module = self.candidate_residual_flow
        if module is None:
            return output
        if module.use_improvement_gated:
            # JFER-v2 fixes guarded membership on the unmodified 12-bank and
            # applies geometry only after the final six identities are known.
            return output
        world_hidden = output.get("protected_expansion_world_hidden")
        if world_hidden is None:
            raise RuntimeError(
                "Candidate residual flow requires all-bank world features"
            )

        trajectories = output["joint_trajs"]
        logits = output["joint_logits"]
        selection_logits = output[
            "protected_expansion_selection_logits"
        ]
        branch_ids = torch.arange(
            logits.shape[1], device=logits.device
        )[None].expand(logits.shape[0], -1)
        branch_ids = branch_ids >= self.num_output_modes
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=trajectories,
            input_dict=input_dict,
            measurement_steps=module.measurement_steps,
            dt=self.dt,
        ).type_as(world_hidden)

        formal_stage = (
            self.train_stage
            == "candidate_bank_residual_flow_full_convergence"
        )
        formal_phase = (
            self._jfer_full_convergence_phase(input_dict)
            if formal_stage and self.training
            else 3
        )
        detach_inputs = self.train_stage in {
            "candidate_bank_residual_flow_geometry",
            "candidate_bank_residual_flow_score",
            "candidate_bank_residual_flow_warmup",
        } or (formal_stage and formal_phase == 0)
        detach_inputs = bool(
            self.candidate_residual_flow_cfg.get(
                "DETACH_INPUTS", detach_inputs
            )
        )
        if detach_inputs:
            trajectories = trajectories.detach()
            logits = logits.detach()
            selection_logits = selection_logits.detach()
            world_hidden = world_hidden.detach()
            spatial_evidence = spatial_evidence.detach()
            scene_context = state["scene_token"].detach()
            pair_state = state["pair_state"].detach()
        elif formal_stage and formal_phase == 1:
            # Phase B trains the existing confidence/selection paths but keeps
            # trajectory, world, scene, and support geometry fixed.
            trajectories = trajectories.detach()
            world_hidden = world_hidden.detach()
            spatial_evidence = spatial_evidence.detach()
            scene_context = state["scene_token"].detach()
            pair_state = state["pair_state"].detach()
        else:
            scene_context = state["scene_token"]
            pair_state = state["pair_state"]

        enable_flow_scoring = bool(
            self.candidate_residual_flow_cfg.get(
                "ENABLE_SCORING",
                self.train_stage
                not in {
                    "candidate_bank_residual_flow_geometry",
                    "candidate_bank_residual_flow_generation_joint",
                },
            )
        )
        pair_type_ids = self._pair_type_ids(
            input_dict, logits.device, logits.shape[0]
        )
        flow_output = module(
            world_hidden=world_hidden,
            trajectories=trajectories,
            base_logits=logits,
            selection_logits=selection_logits,
            branch_ids=branch_ids,
            pair_type_ids=pair_type_ids,
            scene_context=scene_context,
            spatial_evidence=spatial_evidence,
            pair_state=pair_state,
            enable_scoring=enable_flow_scoring,
        )
        reciprocal = bool(
            self.candidate_expansion is not None
            and getattr(
                self.candidate_expansion,
                "is_reciprocal_conditional",
                False,
            )
        )
        preserve_reciprocal = bool(
            reciprocal
            and not self.candidate_residual_flow_cfg.get(
                "APPLY_TO_RECIPROCAL_EXPANSIONS", False
            )
        )
        refined_trajectories = flow_output["trajectories"]
        if preserve_reciprocal:
            protected = self.num_output_modes
            expansion_trajectories = output["joint_trajs"][:, protected:]
            donor_indices = output["candidate_expansion_donor_indices"]
            refined_parent = WorldConditionedJointCandidateBank._gather_modes(
                refined_trajectories[:, :protected], donor_indices
            )
            influencer_indices = output[
                "candidate_expansion_influencer_indices"
            ].long()
            influencer_mask = F.one_hot(
                influencer_indices, num_classes=2
            ).type_as(expansion_trajectories)
            influencer_mask = influencer_mask[
                None, :, :, None, None
            ]
            expansion_trajectories = (
                expansion_trajectories * (1.0 - influencer_mask)
                + refined_parent * influencer_mask
            )
            refined_trajectories = torch.cat(
                [
                    refined_trajectories[:, :protected],
                    expansion_trajectories,
                ],
                dim=1,
            )
        admission = output.get("protected_expansion_admission_logits")
        anchor_admission = admission
        if admission is not None:
            num_expansion = admission.shape[1]
            anchor_admission = (
                admission
                + flow_output["admission_delta"][:, -num_expansion:]
            )

        b2_output = None
        wcpr_output = None
        fixed_e2_indices = None
        deployed_logits = flow_output["joint_logits"]
        deployed_selection_logits = flow_output["selection_logits"]
        if self.jfer_b2_confidence is not None:
            if anchor_admission is None:
                raise RuntimeError(
                    "JFER B2 requires protected expansion admission logits"
                )
            anchor_output = dict(output)
            anchor_output["joint_trajs"] = refined_trajectories
            anchor_output["joint_logits"] = flow_output["joint_logits"]
            anchor_output["protected_expansion_selection_logits"] = (
                flow_output["selection_logits"]
            )
            anchor_output["protected_expansion_admission_logits"] = (
                anchor_admission
            )
            fixed_e2_indices = self._guarded_expansion_indices(anchor_output)
            selected_mask = torch.zeros_like(
                flow_output["joint_logits"], dtype=torch.bool
            )
            selected_mask.scatter_(1, fixed_e2_indices, True)
            protected = self.num_output_modes
            full_admission = flow_output["joint_logits"].new_full(
                flow_output["joint_logits"].shape, 20.0
            )
            full_admission[:, protected:] = anchor_admission
            b2_output = self.jfer_b2_confidence(
                candidate_hidden=flow_output["candidate_hidden"],
                horizon_hidden=flow_output["horizon_hidden"],
                anchor_logits=flow_output["joint_logits"],
                selection_logits=flow_output["selection_logits"],
                admission_logits=full_admission,
                branch_ids=branch_ids,
                pair_type_ids=pair_type_ids,
                selected_mask=selected_mask,
            )
            deployed_logits = b2_output["deployed_logits"]
            deployed_selection_logits = b2_output[
                "deployed_selection_logits"
            ]
        elif self.wcpr is not None:
            if anchor_admission is None:
                raise RuntimeError("WCPR requires protected admission logits")
            anchor_output = dict(output)
            anchor_output["joint_trajs"] = refined_trajectories
            anchor_output["joint_logits"] = flow_output["joint_logits"]
            anchor_output["protected_expansion_selection_logits"] = (
                flow_output["selection_logits"]
            )
            anchor_output["protected_expansion_admission_logits"] = (
                anchor_admission
            )
            fixed_e2_indices = self._guarded_expansion_indices(anchor_output)
            selected_mask = torch.zeros_like(
                flow_output["joint_logits"], dtype=torch.bool
            )
            selected_mask.scatter_(1, fixed_e2_indices, True)
            protected = self.num_output_modes
            full_admission = flow_output["joint_logits"].new_full(
                flow_output["joint_logits"].shape, 20.0
            )
            full_admission[:, protected:] = anchor_admission
            local_trajectories = torch.stack(
                [
                    refined_trajectories[:, :, 0],
                    self._anchor0_to_agent1(
                        refined_trajectories[:, :, 1],
                        state["pair_center_world"],
                    ),
                ],
                dim=2,
            )
            context = state.get("wcpr_context")
            if context is None:
                raise RuntimeError("WCPR encoded context was not prepared")
            wcpr_output = self.wcpr(
                canonical_trajectories=refined_trajectories,
                local_trajectories=local_trajectories,
                pair_state=pair_state,
                world_hidden=flow_output["context"]["world_hidden"],
                scene_context=scene_context,
                candidate_hidden=flow_output["candidate_hidden"],
                horizon_hidden=flow_output["horizon_hidden"],
                anchor_logits=flow_output["joint_logits"],
                selection_logits=flow_output["selection_logits"],
                admission_logits=full_admission,
                branch_ids=branch_ids,
                pair_type_ids=pair_type_ids,
                selected_mask=selected_mask,
                **context,
            )
            deployed_logits = wcpr_output["deployed_logits"]
            deployed_selection_logits = wcpr_output[
                "deployed_selection_logits"
            ]

        refined = dict(output)
        refined["joint_trajs"] = refined_trajectories
        refined["joint_logits"] = deployed_logits
        refined["protected_expansion_selection_logits"] = (
            deployed_selection_logits
        )
        if anchor_admission is not None:
            refined["protected_expansion_admission_logits"] = anchor_admission
        refined["candidate_residual_flow"] = flow_output
        refined["candidate_residual_flow_spatial_evidence"] = spatial_evidence
        if b2_output is not None:
            refined["jfer_b2_confidence"] = b2_output
            refined["jfer_b2_fixed_e2_indices"] = fixed_e2_indices
        if wcpr_output is not None:
            refined["wcpr"] = wcpr_output
            refined["wcpr_fixed_e2_indices"] = fixed_e2_indices
        return refined

    def _apply_cwrr(self, output, state, input_dict):
        if self.cwrr is None:
            return output
        world_hidden = output.get("protected_expansion_world_hidden")
        context = state.get("cwrr_context")
        if world_hidden is None or context is None:
            raise RuntimeError(
                "CWRR requires frozen 12-bank world/context tensors"
            )
        trajectories = output["joint_trajs"].detach()
        logits = output["joint_logits"].detach()
        selection_logits = output[
            "protected_expansion_selection_logits"
        ].detach()
        response = self.cwrr(
            trajectories=trajectories,
            world_hidden=world_hidden.detach(),
            scene_context=state["scene_token"].detach(),
            context=context,
        )
        selected = self._guarded_expansion_indices(output).detach()
        batch = torch.arange(selected.shape[0], device=selected.device)[:, None]
        conditioned = dict(output)
        conditioned["cwrr_response"] = response
        conditioned["cwrr_selected_indices"] = selected
        conditioned["cwrr_frozen_candidate_trajs"] = trajectories
        conditioned["cwrr_frozen_candidate_logits"] = logits
        conditioned["cwrr_frozen_selection_logits"] = selection_logits
        if self.cwrr.enable_refinement:
            selected_trajs = trajectories[batch, selected]
            selected_logits = logits[batch, selected]
            selected_world = world_hidden.detach()[batch, selected]
            response_world = response["world"][batch, selected]
            if self.cwrr.variant == "none":
                response_world = torch.zeros_like(response_world)
            refinement = self.cwrr.refiner(
                trajectories=selected_trajs,
                response_world=response_world,
                candidate_world=selected_world,
                enabled=True,
            )
            refinement["base_trajectories"] = selected_trajs
            conditioned["cwrr_refinement"] = refinement
            conditioned["cwrr_refinement_logits"] = selected_logits
        return conditioned

    def _append_reciprocal_augmentation(self, output, state, input_dict):
        generator = self.reciprocal_candidate_expansion
        selector_module = self.reciprocal_set_selector
        joint_decoder = self.reciprocal_world_set_decoder
        if generator is None or (
            selector_module is None and joint_decoder is None
        ):
            return output
        if output["joint_logits"].shape[1] != (
            self.num_output_modes + self.num_expansion_modes
        ):
            raise RuntimeError(
                "Reciprocal augmentation expected the complete legacy JFER "
                "candidate bank"
            )

        # Freeze the exact six-mode set that the verified policy would deploy.
        # Reciprocal proposals are generated after this decision and therefore
        # cannot alter the legacy set through candidate-set attention or NMS.
        legacy_indices = self._guarded_expansion_indices(output)
        batch_size = legacy_indices.shape[0]
        batch_index = torch.arange(
            batch_size, device=legacy_indices.device
        )[:, None]
        legacy_trajs = output["joint_trajs"]
        legacy_logits = output["joint_logits"]
        legacy_world_hidden = output.get("protected_expansion_world_hidden")
        expected_hidden_shape = (
            legacy_trajs.shape[0],
            legacy_trajs.shape[1],
            generator.num_horizons,
            self.hidden_dim,
        )
        if (
            legacy_world_hidden is None
            or legacy_world_hidden.ndim != 4
            or tuple(legacy_world_hidden.shape) != expected_hidden_shape
        ):
            raise RuntimeError(
                "Reciprocal augmentation requires legacy JFER horizon tokens "
                f"with shape {expected_hidden_shape}, got "
                f"{None if legacy_world_hidden is None else tuple(legacy_world_hidden.shape)}"
            )
        base_trajs = legacy_trajs[batch_index, legacy_indices].detach()
        base_logits = legacy_logits[batch_index, legacy_indices].detach()
        base_horizon_hidden = legacy_world_hidden[
            batch_index, legacy_indices
        ].detach()
        base_hidden = base_horizon_hidden.mean(dim=2)

        donor_indices = generator.select_donor_indices(base_logits)
        donor_hidden = WorldConditionedJointCandidateBank._gather_modes(
            base_hidden, donor_indices
        )
        donor_logits = base_logits.gather(1, donor_indices)
        donor_trajs = WorldConditionedJointCandidateBank._gather_modes(
            base_trajs, donor_indices
        )
        expansion = generator(
            pair_hidden=donor_hidden,
            pair_endpoints=donor_trajs[..., -1, :],
            donor_hidden=donor_hidden,
            donor_logits=donor_logits,
            donor_trajs=donor_trajs,
            proposal_trajs=None,
            proposal_prior=donor_logits.new_zeros(donor_logits.shape),
            base_hidden=base_hidden,
            base_logits=base_logits,
            base_trajs=base_trajs,
        )

        detach_selector_input = bool(
            self.reciprocal_augmentation_cfg.get(
                "DETACH_GENERATOR_FOR_SELECTOR", True
            )
        )

        def selector_input(value):
            return value.detach() if detach_selector_input else value

        expansion_hidden = expansion["horizon_token"].mean(dim=2)
        horizon_evidence = expansion["horizon_logits"]
        signed_horizon_evidence = (
            2.0 * torch.sigmoid(horizon_evidence) - 1.0
        )
        scene_memory = state.get("scene_memory")
        world_rollout = state.get("prior_rollout")
        world_logits = state.get("world_logits")
        if torch.is_tensor(scene_memory):
            scene_memory = scene_memory.detach()
        if torch.is_tensor(world_rollout):
            world_rollout = world_rollout.detach()
        if torch.is_tensor(world_logits):
            world_logits = world_logits.detach()
        selected_shifts = base_logits.new_zeros(
            batch_size, self.num_output_modes, 2
        )
        pair_type_ids = self._pair_type_ids(
            input_dict, base_logits.device, batch_size
        )
        replacement = None
        if selector_module is not None:
            replacement = selector_module(
                base_hidden=base_hidden,
                expansion_hidden=selector_input(expansion_hidden),
                base_trajectories=base_trajs,
                expansion_trajectories=selector_input(
                    expansion["joint_trajs"]
                ),
                base_logits=base_logits,
                donor_indices=donor_indices,
                selected_shifts=selected_shifts,
                frozen_horizon_logits=selector_input(horizon_evidence),
                frozen_gain=selector_input(signed_horizon_evidence),
                scene_context=state["scene_token"].detach(),
                pair_type_ids=pair_type_ids,
                scene_memory=scene_memory,
                world_rollout=world_rollout,
                world_logits=world_logits,
                world_sample_indices=self.sample_indices,
            )
        joint_decoder_output = None
        if joint_decoder is not None:
            detach_joint_input = bool(
                self.reciprocal_augmentation_cfg.get(
                    "DETACH_GENERATOR_FOR_JOINT_DECODER", True
                )
            )

            def joint_input(value):
                return value.detach() if detach_joint_input else value

            joint_decoder_output = joint_decoder(
                base_hidden=base_hidden,
                expansion_hidden=joint_input(expansion_hidden),
                base_horizon_hidden=base_horizon_hidden,
                expansion_horizon_hidden=joint_input(
                    expansion["horizon_token"]
                ),
                base_trajectories=base_trajs,
                expansion_trajectories=joint_input(
                    expansion["joint_trajs"]
                ),
                base_logits=base_logits,
                expansion_logits=joint_input(expansion["joint_logits"]),
                scene_context=state["scene_token"].detach(),
                pair_type_ids=pair_type_ids,
            )

        augmented = dict(output)
        augmented["legacy_jfer_joint_trajs"] = legacy_trajs
        augmented["legacy_jfer_joint_logits"] = legacy_logits
        augmented["legacy_jfer_selection_logits"] = output[
            "protected_expansion_selection_logits"
        ]
        augmented["legacy_jfer_admission_logits"] = output[
            "protected_expansion_admission_logits"
        ]
        augmented["legacy_jfer_world_hidden"] = legacy_world_hidden
        augmented["legacy_jfer_deployed_indices"] = legacy_indices
        augmented["protected_base_joint_trajs"] = base_trajs
        augmented["protected_base_joint_logits"] = base_logits
        augmented["joint_trajs"] = torch.cat(
            [base_trajs, expansion["joint_trajs"]], dim=1
        )
        augmented["joint_logits"] = torch.cat(
            [base_logits, expansion["joint_logits"]], dim=1
        )
        augmented["protected_expansion_selection_logits"] = torch.cat(
            [base_logits, expansion["selection_logits"]], dim=1
        )
        augmented["protected_expansion_admission_logits"] = expansion[
            "admission_logits"
        ]
        augmented["protected_expansion_candidate_selector_logits"] = (
            expansion["candidate_selector_logits"]
        )
        augmented["protected_expansion_scene_gate_logits"] = expansion[
            "scene_replacement_gate_logits"
        ]
        augmented["protected_expansion_replacement_utility"] = expansion[
            "replacement_utility"
        ]
        augmented["protected_expansion_world_hidden"] = torch.cat(
            [base_horizon_hidden, expansion["horizon_token"]], dim=1
        )
        augmented["candidate_expansion_horizon_logits"] = expansion[
            "horizon_logits"
        ]
        augmented["candidate_expansion_score_delta"] = expansion[
            "score_delta"
        ]
        augmented["candidate_expansion_waypoint_delta"] = expansion[
            "waypoint_delta"
        ]
        augmented["candidate_expansion_dense_knot_delta"] = expansion[
            "dense_knot_delta"
        ]
        augmented["candidate_expansion_dense_temporal_delta"] = expansion[
            "dense_temporal_delta"
        ]
        augmented["candidate_expansion_goal_gate"] = expansion["goal_gate"]
        augmented["candidate_expansion_confidence_gate"] = expansion[
            "confidence_gate"
        ]
        augmented["candidate_expansion_unified_confidence_residual"] = (
            expansion["unified_confidence_residual"]
        )
        augmented["candidate_expansion_unified_confidence_temperature"] = (
            expansion["unified_confidence_temperature"]
        )
        augmented["candidate_expansion_pair_indices"] = donor_indices
        augmented["candidate_expansion_donor_indices"] = donor_indices
        augmented["candidate_expansion_influencer_indices"] = expansion[
            "influencer_indices"
        ]
        augmented["candidate_expansion_reactor_indices"] = expansion[
            "reactor_indices"
        ]
        augmented["candidate_expansion_interaction_probability"] = expansion[
            "interaction_probability"
        ]
        reciprocal_output = dict(expansion)
        reciprocal_output["trajectories"] = expansion["joint_trajs"]
        reciprocal_output["donor_indices"] = donor_indices
        reciprocal_output["selected_shifts"] = selected_shifts
        reciprocal_output["set_marginal_replacement"] = replacement
        reciprocal_output["world_set_decoder"] = joint_decoder_output
        augmented["reciprocal_augmentation"] = reciprocal_output
        if replacement is not None:
            augmented["reciprocal_set_replacement"] = replacement
        if joint_decoder_output is not None:
            augmented["reciprocal_world_set_decoder"] = joint_decoder_output
        return augmented

    def _apply_deployed_set_scorer(self, output, state, input_dict):
        if self.deployed_set_scorer is None:
            return output
        if "protected_expansion_world_hidden" not in output:
            raise RuntimeError(
                "Deployed-set scoring requires protected expansion features"
            )

        # Membership is decided once from the frozen stage-one selector.  The
        # learned scorer can reorder these six modes but cannot admit a new one.
        selected = self._guarded_expansion_indices(output)
        batch_size = selected.shape[0]
        batch_idx = torch.arange(
            batch_size, device=selected.device
        )[:, None]
        selected_trajs = output["joint_trajs"][batch_idx, selected].detach()
        selected_logits = output["joint_logits"][
            batch_idx, selected
        ].detach()
        selected_selection_logits = output[
            "protected_expansion_selection_logits"
        ][batch_idx, selected].detach()
        selected_world_hidden = output[
            "protected_expansion_world_hidden"
        ][batch_idx, selected].detach()
        branch_ids = (selected >= self.num_output_modes).long()
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=selected_trajs,
            input_dict=input_dict,
            measurement_steps=self.deployed_set_scorer.measurement_steps,
            dt=self.dt,
        ).type_as(selected_world_hidden)
        pair_type_ids = self._pair_type_ids(
            input_dict,
            selected.device,
            batch_size,
        )
        scorer_output = self.deployed_set_scorer(
            world_hidden=selected_world_hidden,
            trajectories=selected_trajs,
            base_logits=selected_logits,
            selection_logits=selected_selection_logits,
            branch_ids=branch_ids,
            pair_type_ids=pair_type_ids,
            scene_context=state["scene_token"].detach(),
            spatial_evidence=spatial_evidence.detach(),
        )

        rescored_logits = output["joint_logits"].scatter(
            1, selected, scorer_output["joint_logits"]
        )
        rescored = dict(output)
        rescored["joint_logits"] = rescored_logits
        rescored["deployed_set_scorer"] = scorer_output
        rescored["deployed_set_indices"] = selected
        rescored["deployed_set_trajs"] = selected_trajs
        rescored["deployed_set_spatial_evidence"] = spatial_evidence
        return rescored

    def _apply_deployed_trajectory_refiner(
        self, output, state, input_dict
    ):
        refiner = self.deployed_trajectory_refiner
        if refiner is None:
            return output
        if "protected_expansion_world_hidden" not in output:
            raise RuntimeError(
                "Deployed trajectory refinement requires protected "
                "expansion features"
            )

        # Membership and confidence are fixed before any geometric update.
        # The same indices and logits are reused by select_final, so trajectory
        # refinement cannot silently alter admission, NMS, or score ranking.
        selected = self._guarded_expansion_indices(output)
        batch_size = selected.shape[0]
        batch_idx = torch.arange(
            batch_size, device=selected.device
        )[:, None]
        selected_trajs = output["joint_trajs"][
            batch_idx, selected
        ].detach()
        selected_logits = output["joint_logits"][
            batch_idx, selected
        ].detach()
        selected_selection_logits = output[
            "protected_expansion_selection_logits"
        ][batch_idx, selected].detach()
        selected_world_hidden = output[
            "protected_expansion_world_hidden"
        ][batch_idx, selected].detach()
        branch_ids = (selected >= self.num_output_modes).long()
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=selected_trajs,
            input_dict=input_dict,
            measurement_steps=refiner.measurement_steps,
            dt=self.dt,
        ).type_as(selected_world_hidden)
        refiner_output = refiner(
            world_hidden=selected_world_hidden,
            trajectories=selected_trajs,
            logits=selected_logits,
            selection_logits=selected_selection_logits,
            branch_ids=branch_ids,
            pair_type_ids=self._pair_type_ids(
                input_dict,
                selected.device,
                batch_size,
            ),
            scene_context=state["scene_token"].detach(),
            spatial_evidence=spatial_evidence.detach(),
        )
        refined = dict(output)
        refined["deployed_refiner_indices"] = selected
        refined["deployed_refiner_logits"] = selected_logits
        refined["deployed_refiner_selection_logits"] = (
            selected_selection_logits
        )
        refined["deployed_refiner_output"] = refiner_output
        refined["deployed_refiner_spatial_evidence"] = spatial_evidence
        return refined

    def _apply_improvement_gated_residual_flow(
        self, output, state, input_dict
    ):
        module = self.candidate_residual_flow
        if not self.use_improvement_gated_residual_flow:
            return output
        world_hidden = output.get("protected_expansion_world_hidden")
        if world_hidden is None:
            raise RuntimeError(
                "Improvement-gated JFER requires protected world features"
            )

        # Guarded selection, score and order are sealed before refinement.
        selected = self._guarded_expansion_indices(output)
        batch_size = selected.shape[0]
        batch_idx = torch.arange(
            batch_size, device=selected.device
        )[:, None]
        selected_trajs = output["joint_trajs"][batch_idx, selected].detach()
        selected_logits = output["joint_logits"][batch_idx, selected].detach()
        selected_selection_logits = output[
            "protected_expansion_selection_logits"
        ][batch_idx, selected].detach()
        selected_world_hidden = world_hidden[batch_idx, selected].detach()
        branch_ids = (selected >= self.num_output_modes).long()
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=selected_trajs,
            input_dict=input_dict,
            measurement_steps=module.measurement_steps,
            dt=self.dt,
        ).type_as(selected_world_hidden).detach()
        phase = (
            module.improvement_phase(input_dict["cur_epoch"])
            if "cur_epoch" in input_dict
            else 2
        )
        improvement_output = module.forward_improvement_gated(
            world_hidden=selected_world_hidden,
            trajectories=selected_trajs,
            base_logits=selected_logits,
            selection_logits=selected_selection_logits,
            branch_ids=branch_ids,
            pair_type_ids=self._pair_type_ids(
                input_dict, selected.device, batch_size
            ),
            scene_context=state["scene_token"].detach(),
            spatial_evidence=spatial_evidence,
            pair_state=state["pair_state"].detach(),
            phase=phase,
        )
        refined = dict(output)
        refined["improvement_gated_indices"] = selected
        refined["improvement_gated_logits"] = selected_logits
        refined["improvement_gated_selection_logits"] = (
            selected_selection_logits
        )
        refined["improvement_gated_output"] = improvement_output
        refined["improvement_gated_spatial_evidence"] = spatial_evidence
        return refined

    def _apply_set_to_slot_transport(self, output, state, input_dict):
        transport = self.set_to_slot_transport
        if transport is None:
            return output
        world_hidden = output.get("protected_expansion_world_hidden")
        if world_hidden is None:
            raise RuntimeError(
                "Set-to-slot transport requires all-bank world features"
            )

        # The guarded policy is the exact zero-step assignment.  The learned
        # module may replace any slot, but each slot retains its trusted score
        # value and the six-score multiset is therefore invariant.
        baseline_indices = self._guarded_expansion_indices(output)
        num_candidates = output["joint_logits"].shape[1]
        branch_ids = torch.arange(
            num_candidates, device=output["joint_logits"].device
        )[None].expand(output["joint_logits"].shape[0], -1)
        branch_ids = branch_ids >= self.num_output_modes
        transport_output = transport(
            world_hidden=world_hidden,
            trajectories=output["joint_trajs"],
            base_logits=output["joint_logits"],
            selection_logits=output[
                "protected_expansion_selection_logits"
            ],
            branch_ids=branch_ids,
            pair_type_ids=self._pair_type_ids(
                input_dict,
                output["joint_logits"].device,
                output["joint_logits"].shape[0],
            ),
            scene_context=state["scene_token"],
            baseline_indices=baseline_indices,
        )
        transported = dict(output)
        transported["set_to_slot_transport"] = transport_output
        return transported

    def _apply_sequential_mode_decoder(self, output, state, input_dict):
        module = self.sequential_mode_decoder
        if module is None:
            return output
        world_hidden = output.get("protected_expansion_world_hidden")
        if world_hidden is None:
            raise RuntimeError(
                "Sequential mode decoding requires all-bank world features"
            )
        context = state.get("sequential_mode_context")
        if context is None:
            raise RuntimeError("Sequential mode context was not prepared")
        trajectories = output["joint_trajs"]
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=trajectories,
            input_dict=input_dict,
            measurement_steps=module.measurement_steps,
            dt=self.dt,
        ).type_as(world_hidden)
        num_candidates = trajectories.shape[1]
        branch_ids = (
            torch.arange(num_candidates, device=trajectories.device)[None]
            .expand(trajectories.shape[0], -1)
            .ge(self.num_output_modes)
            .long()
        )
        sequential_output = module(
            world_hidden=world_hidden.detach(),
            trajectories=trajectories.detach(),
            base_logits=output["joint_logits"].detach(),
            selection_logits=output[
                "protected_expansion_selection_logits"
            ].detach(),
            branch_ids=branch_ids,
            pair_type_ids=self._pair_type_ids(
                input_dict,
                trajectories.device,
                trajectories.shape[0],
            ),
            scene_context=state["scene_token"].detach(),
            spatial_evidence=spatial_evidence.detach(),
            baseline_indices=self._guarded_expansion_indices(output),
            context=context,
        )
        conditioned = dict(output)
        conditioned["sequential_mode_decoder"] = sequential_output
        conditioned["sequential_mode_spatial_evidence"] = spatial_evidence
        return conditioned

    @staticmethod
    def _apply_single_confidence_preserving_replacement(
        base_trajectories, expansion_trajectories, replacement
    ):
        selected = base_trajectories.clone()
        accepted_rows = replacement["accepted"].nonzero(
            as_tuple=False
        ).flatten()
        if accepted_rows.numel() > 0:
            selected[
                accepted_rows,
                replacement["slot_index"][accepted_rows],
            ] = expansion_trajectories[
                accepted_rows,
                replacement["expansion_index"][accepted_rows],
            ]
        return selected

    def select_final(self, final_output):
        cwrr_refinement = final_output.get("cwrr_refinement")
        if cwrr_refinement is not None:
            return (
                torch.softmax(
                    final_output["cwrr_refinement_logits"], dim=-1
                ),
                cwrr_refinement["trajectories"],
            )
        reciprocal = final_output.get("reciprocal_augmentation")
        replacement = final_output.get("reciprocal_set_replacement")
        joint_decoder = final_output.get("reciprocal_world_set_decoder")
        if joint_decoder is not None:
            return (
                torch.softmax(joint_decoder["final_logits"], dim=-1),
                joint_decoder["final_trajectories"],
            )
        if getattr(self, "use_reciprocal_augmentation", False) and (
            reciprocal is None
            or (
                self.reciprocal_world_set_decoder is not None
                and joint_decoder is None
            )
            or (
                self.reciprocal_world_set_decoder is None
                and replacement is None
            )
        ):
            raise RuntimeError(
                "Configured reciprocal augmentation is incomplete; refusing "
                "to fall through to a different deployment policy"
            )
        if (
            self.reciprocal_world_set_decoder is None
            and (reciprocal is None) != (replacement is None)
        ):
            raise RuntimeError(
                "Reciprocal candidates and replacement decisions must be "
                "present together"
            )
        if reciprocal is not None and replacement is not None:
            base_logits = final_output["protected_base_joint_logits"]
            base_trajs = final_output["protected_base_joint_trajs"]
            selected_trajs = (
                self._apply_single_confidence_preserving_replacement(
                    base_trajs,
                    reciprocal["trajectories"],
                    replacement,
                )
            )
            # The trajectory assigned to one slot may change; the trusted six
            # score values and their normalization remain bitwise unchanged.
            return torch.softmax(base_logits, dim=-1), selected_trajs
        sequential = final_output.get("sequential_mode_decoder")
        if sequential is not None:
            return (
                torch.softmax(sequential["slot_logits"], dim=-1),
                sequential["refined_trajectories"],
            )
        if "world_credit_posterior" in final_output:
            logits = final_output["world_credit_posterior"]["joint_logits"]
            trajectories = final_output["world_credit_selected_trajs"]
            if logits.shape[1] != self.num_output_modes:
                raise RuntimeError(
                    "World-credit deployment support must contain exactly "
                    f"{self.num_output_modes} modes, got {logits.shape[1]}"
                )
            selected = logits.argsort(dim=-1, descending=True)
            batch_index = torch.arange(
                logits.shape[0], device=logits.device
            )[:, None]
            selected_logits = logits[batch_index, selected]
            selected_trajectories = trajectories[batch_index, selected]
            return (
                torch.softmax(selected_logits, dim=-1),
                selected_trajectories,
            )
        time_warp = final_output.get("interaction_time_warp_bank")
        replacement = (
            time_warp.get("set_marginal_replacement")
            if time_warp is not None
            else None
        )
        if replacement is not None:
            base_logits = final_output["protected_base_joint_logits"]
            base_trajs = final_output["protected_base_joint_trajs"]
            expansion_trajs = time_warp["trajectories"]
            selected_trajs = (
                self._apply_single_confidence_preserving_replacement(
                    base_trajs, expansion_trajs, replacement
                )
            )
            # The trajectory occupying a slot may change; the score attached
            # to that slot never does. This is the deployment invariant used
            # by the 37-action training target.
            return torch.softmax(base_logits, dim=-1), selected_trajs
        transport_output = final_output.get("set_to_slot_transport")
        if transport_output is not None:
            return (
                torch.softmax(transport_output["slot_logits"], dim=-1),
                transport_output["selected_trajectories"],
            )
        refiner_output = final_output.get("deployed_refiner_output")
        if refiner_output is not None:
            # Scores and selected identities are the frozen pre-refinement
            # values.  Only trajectory coordinates can differ.
            logits = final_output["deployed_refiner_logits"]
            return (
                torch.softmax(logits, dim=-1),
                refiner_output["trajectories"],
            )
        improvement_output = final_output.get("improvement_gated_output")
        if improvement_output is not None:
            # The six logits and their order were frozen before geometry.
            logits = final_output["improvement_gated_logits"]
            return (
                torch.softmax(logits, dim=-1),
                improvement_output["trajectories"],
            )
        return super().select_final(final_output)

    def condition_predictions(
        self,
        layer_idx,
        query_content,
        pred_scores,
        pred_trajs,
        state,
        input_dict,
    ):
        batch_size = state["batch_size"]
        num_modes = pred_scores.shape[1]
        marginal_log_prior = torch.log_softmax(
            pred_scores.reshape(batch_size, 2, num_modes),
            dim=-1,
        ).sum(dim=1)
        output = super().condition_predictions(
            layer_idx=layer_idx,
            query_content=query_content,
            pred_scores=pred_scores,
            pred_trajs=pred_trajs,
            state=state,
            input_dict=input_dict,
        )

        if self.structured_behavior_world is not None:
            pair_center_world = input_dict[
                "pair_center_objects_world"
            ].to(pred_trajs.device).type_as(pred_trajs)
            output, state = self.structured_behavior_world.condition_output(
                layer_idx=layer_idx,
                output=output,
                state=state,
                pair_center_world=pair_center_world,
                to_local=self._to_local,
                gather_center_modes=self._gather_center_modes,
            )

        if layer_idx + 1 < self.num_decoder_layers:
            bank_prior = state["candidate_bank_log_prior"]
            state["candidate_bank_log_prior"] = bank_prior.gather(
                1, output["mode_order"]
            )

        if (
            self.metric_scorer is not None
            and layer_idx + 1 == self.num_decoder_layers
            and (
                not bool(
                    output.get(
                        "structured_world_owns_final_probability", False
                    )
                )
                or bool(
                    getattr(
                        self.structured_behavior_world,
                        "use_unified_success_posterior",
                        False,
                    )
                )
            )
        ):
            spatial_evidence = None
            if self.metric_scorer.use_spatial_evidence:
                spatial_evidence = build_candidate_spatial_evidence(
                    canonical_trajectories=output["joint_trajs"],
                    input_dict=input_dict,
                    measurement_steps=self.metric_scorer.measurement_steps,
                    dt=self.dt,
                )
            detach_inputs = self.train_stage in {
                "candidate_bank_ap_calibration",
                "candidate_bank_permutation_calibration",
            }
            scorer_inputs = {
                "world_response": output["response"],
                "trajectories": output["joint_trajs"],
                "base_logits": output["joint_logits"],
                "marginal_log_prior": marginal_log_prior,
                "bank_log_prior": state["candidate_bank_log_prior"],
                "joint_context": state["joint_token"],
                "scene_context": state["scene_token"],
                "spatial_evidence": spatial_evidence,
            }
            if detach_inputs:
                scorer_inputs = {
                    key: (
                        value.detach()
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in scorer_inputs.items()
                }
            scorer_output = self.metric_scorer(
                **scorer_inputs,
                pair_type_ids=self._pair_type_ids(
                    input_dict,
                    output["joint_logits"].device,
                    batch_size,
                ),
            )
            output["joint_pre_metric_logits"] = output["joint_logits"]
            output["joint_logits"] = scorer_output["joint_logits"]
            output["metric_aligned_scorer"] = scorer_output
            if spatial_evidence is not None:
                output["candidate_spatial_evidence"] = spatial_evidence
        if (
            self.use_candidate_expansion
            and layer_idx + 1 == self.num_decoder_layers
        ):
            if self.interaction_time_warp_bank is not None:
                output = self._append_interaction_time_warp_expansions(
                    output, state, input_dict
                )
            else:
                output = self._append_candidate_expansions(output, state)
            if self.sequential_mode_decoder is not None:
                output = self._apply_sequential_mode_decoder(
                    output, state, input_dict
                )
            if self.candidate_residual_flow is not None:
                output = self._apply_candidate_residual_flow(
                    output, state, input_dict
                )
            if self.use_improvement_gated_residual_flow:
                output = self._apply_improvement_gated_residual_flow(
                    output, state, input_dict
                )
            if self.cwrr is not None:
                output = self._apply_cwrr(output, state, input_dict)
            if self.deployed_set_scorer is not None:
                output = self._apply_deployed_set_scorer(
                    output, state, input_dict
                )
            if self.deployed_trajectory_refiner is not None:
                output = self._apply_deployed_trajectory_refiner(
                    output, state, input_dict
                )
            if self.set_to_slot_transport is not None:
                output = self._apply_set_to_slot_transport(
                    output, state, input_dict
                )
            if self.reciprocal_candidate_expansion is not None:
                output = self._append_reciprocal_augmentation(
                    output, state, input_dict
                )
        if (
            layer_idx + 1 == self.num_decoder_layers
            and bool(
                output.get("structured_world_owns_final_probability", False)
            )
        ):
            condition_expanded = getattr(
                self.structured_behavior_world,
                "condition_expanded_output",
                None,
            )
            if condition_expanded is None:
                raise RuntimeError(
                    "Structured world claimed the final probability without "
                    "implementing condition_expanded_output"
                )
            # Candidate membership is frozen before the learned posterior sees
            # the set. This preserves the verified Coverage Transport geometry
            # while allowing world evidence to correct confidence ordering.
            if self.use_candidate_expansion:
                output["world_credit_selected_indices"] = (
                    self._guarded_expansion_indices(output)
                )
            output, state = condition_expanded(
                output, state, input_dict=input_dict
            )
        return output

    def _global_ap_surrogate(
        self,
        probability,
        credit,
        group_ids,
        scorer_cfg=None,
    ):
        score = probability[..., None].expand(
            -1, -1, credit["credited_target"].shape[-1]
        )
        gathered_score = self._gather_with_local_gradient(score)
        gathered_target = self._gather_with_local_gradient(
            credit["credited_target"].to(torch.uint8)
        ).bool()
        gathered_supervision = self._gather_with_local_gradient(
            credit["supervision_mask"].to(torch.uint8)
        ).bool()
        gathered_groups = self._gather_with_local_gradient(group_ids)
        if scorer_cfg is None:
            scorer_cfg = self.metric_aligned_scorer_cfg
        temperature = max(
            float(
                scorer_cfg.get(
                    "GLOBAL_AP_TEMPERATURE", 0.05
                )
            ),
            self.eps,
        )
        terms = []
        for group_id in gathered_groups.unique(sorted=True):
            if int(group_id.item()) < 0:
                continue
            group_mask = gathered_groups == group_id
            for horizon_idx in range(gathered_score.shape[-1]):
                positive_mask = (
                    gathered_target[:, :, horizon_idx]
                    & group_mask[:, None]
                )
                negative_mask = (
                    gathered_supervision[:, :, horizon_idx]
                    & ~gathered_target[:, :, horizon_idx]
                    & group_mask[:, None]
                )
                if not bool(positive_mask.any().item()):
                    continue
                if not bool(negative_mask.any().item()):
                    continue
                positive = gathered_score[:, :, horizon_idx][positive_mask]
                negative = gathered_score[:, :, horizon_idx][negative_mask]
                terms.append(
                    F.softplus(
                        (
                            negative[:, None]
                            - positive[None, :]
                        )
                        / temperature
                    ).mean()
                    * temperature
                )
        if not terms:
            return probability.new_zeros(())
        return torch.stack(terms).mean()

    def _bucket_action_utility_ranking_loss(
        self,
        predicted_utility,
        target_utility,
        group_ids,
        selector_cfg,
    ):
        """Rank replacement utility inside official type/behavior buckets."""
        if predicted_utility.shape != target_utility.shape:
            raise ValueError(
                "Predicted and target action utility shapes must match"
            )
        if group_ids.shape != predicted_utility.shape[:1]:
            raise ValueError("One official group id is required per scene")

        gathered_prediction = self._gather_with_local_gradient(
            predicted_utility
        )
        gathered_target = self._gather_with_local_gradient(
            target_utility.detach()
        )
        gathered_group = self._gather_with_local_gradient(group_ids)
        temperature = max(
            float(selector_cfg.get("BUCKET_RANK_TEMPERATURE", 0.05)),
            self.eps,
        )
        target_margin = float(
            selector_cfg.get("BUCKET_RANK_TARGET_MARGIN", 0.005)
        )
        prediction_margin = float(
            selector_cfg.get("BUCKET_RANK_PREDICTION_MARGIN", 0.01)
        )
        max_actions = int(
            selector_cfg.get("BUCKET_RANK_MAX_ACTIONS", 192)
        )
        terms = []
        for group_id in gathered_group.unique(sorted=True):
            if int(group_id.item()) < 0:
                continue
            scene_mask = gathered_group == group_id
            prediction = gathered_prediction[scene_mask].reshape(-1)
            target = gathered_target[scene_mask].reshape(-1)
            if prediction.numel() > max_actions:
                # Keep the most informative positive/negative actions. The
                # selection is target-only and therefore cannot leak gradients.
                keep = target.abs().topk(max_actions).indices
                prediction = prediction[keep]
                target = target[keep]
            target_delta = target[:, None] - target[None, :]
            ordered_pair = target_delta > target_margin
            if not bool(ordered_pair.any().item()):
                continue
            prediction_delta = prediction[:, None] - prediction[None, :]
            terms.append(
                temperature
                * F.softplus(
                    (
                        prediction_margin
                        - prediction_delta[ordered_pair]
                    )
                    / temperature
                ).mean()
            )
        if not terms:
            return predicted_utility.new_zeros(())
        distributed_scale = (
            float(dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else 1.0
        )
        return torch.stack(terms).mean() * distributed_scale

    def _metric_aligned_scorer_loss(self, final_output, input_dict):
        scorer_output = final_output.get("metric_aligned_scorer")
        if scorer_output is None:
            raise RuntimeError(
                "Final output is missing metric-aligned scorer values"
            )
        trajectories = final_output["joint_trajs"].detach()
        metrics = self._official_match_quality(trajectories, input_dict)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=metrics["horizon_cost"].detach(),
            pair_valid=metrics["pair_valid"],
        )
        match_logits = scorer_output["horizon_logits"]
        target = credit["credited_target"].type_as(match_logits)
        supervision = credit["supervision_mask"].type_as(match_logits)
        sample_weights = self._pair_sample_weights(
            input_dict, match_logits.device, match_logits.dtype
        )
        horizon_weights = self.metric_scorer.horizon_weights.type_as(
            match_logits
        )

        focal_alpha = float(
            self.metric_aligned_scorer_cfg.get("FOCAL_ALPHA", 0.75)
        )
        focal_gamma = float(
            self.metric_aligned_scorer_cfg.get("FOCAL_GAMMA", 2.0)
        )
        ce = F.binary_cross_entropy_with_logits(
            match_logits, target, reduction="none"
        )
        match_probability = torch.sigmoid(match_logits)
        p_t = (
            match_probability * target
            + (1.0 - match_probability) * (1.0 - target)
        )
        alpha_t = (
            focal_alpha * target
            + (1.0 - focal_alpha) * (1.0 - target)
        )
        focal = alpha_t * (1.0 - p_t).pow(focal_gamma) * ce
        focal_weight = supervision * horizon_weights[None, None]
        focal_sample = (
            focal * focal_weight
        ).sum(dim=(1, 2)) / focal_weight.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_match = self._weighted_batch_mean(
            focal_sample, sample_weights
        )

        boundary_temperature = max(
            float(
                self.metric_aligned_scorer_cfg.get(
                    "BOUNDARY_TEMPERATURE", 0.2
                )
            ),
            self.eps,
        )
        boundary_target = torch.sigmoid(
            (1.0 - metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        boundary = F.binary_cross_entropy_with_logits(
            match_logits, boundary_target, reduction="none"
        )
        boundary_sample = (
            boundary * focal_weight
        ).sum(dim=(1, 2)) / focal_weight.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_boundary = self._weighted_batch_mean(
            boundary_sample, sample_weights
        )

        horizon_log_probability = F.log_softmax(
            match_logits, dim=1
        )
        listwise = -(
            target * horizon_log_probability
        ).sum(dim=1)
        listwise_weight = (
            credit["has_match"].type_as(match_logits)
            * horizon_weights[None]
        )
        listwise_sample = (
            listwise * listwise_weight
        ).sum(dim=-1) / listwise_weight.sum(
            dim=-1
        ).clamp_min(1.0)
        loss_listwise = self._weighted_batch_mean(
            listwise_sample,
            sample_weights
            * credit["has_match"].any(dim=-1).type_as(match_logits),
        )

        scene_logits = scorer_output["scene_match_logits"]
        scene_target = credit["has_match"].type_as(scene_logits)
        scene_loss = F.binary_cross_entropy_with_logits(
            scene_logits, scene_target, reduction="none"
        )
        scene_sample = (
            scene_loss * horizon_weights[None]
        ).sum(dim=-1)
        loss_scene = self._weighted_batch_mean(
            scene_sample, sample_weights
        )

        final_probability = scorer_output["probability"]
        group_ids = official_ap_group_ids(
            input_dict,
            final_probability.shape[0],
            final_probability.device,
        )
        loss_global_ap = self._global_ap_surrogate(
            final_probability, credit, group_ids
        )

        no_match_weight = (
            (~credit["has_match"] & metrics["pair_valid"])
            .type_as(final_probability)
            * horizon_weights[None]
        ).sum(dim=-1)
        uniform_kl = (
            final_probability
            * torch.log(
                final_probability.clamp_min(self.eps)
                * final_probability.shape[-1]
            )
        ).sum(dim=-1)
        loss_no_match_uniform = self._weighted_batch_mean(
            uniform_kl,
            sample_weights * no_match_weight,
        )

        base_probability = torch.softmax(
            scorer_output["base_logits"].detach(), dim=-1
        )
        prior_kl = (
            base_probability
            * (
                torch.log(base_probability.clamp_min(self.eps))
                - torch.log(final_probability.clamp_min(self.eps))
            )
        ).sum(dim=-1)
        loss_prior_kl = self._weighted_batch_mean(
            prior_kl, sample_weights
        ).clamp_min(0.0)

        loss_permutation_listwise = final_probability.new_zeros(())
        loss_permutation_pairwise = final_probability.new_zeros(())
        loss_permutation_top1 = final_probability.new_zeros(())
        permutation_top1_accuracy = final_probability.new_zeros(())
        permutation_target_margin = final_probability.new_zeros(())
        permutation_utility = scorer_output.get("permutation_utility")
        if permutation_utility is not None:
            valid_horizon_weight = (
                metrics["pair_valid"].type_as(match_logits)
                * horizon_weights[None]
            )
            valid_horizon_denominator = valid_horizon_weight.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.eps)
            credited_utility = (
                target * valid_horizon_weight[:, None]
            ).sum(dim=-1) / valid_horizon_denominator
            boundary_utility = (
                boundary_target * valid_horizon_weight[:, None]
            ).sum(dim=-1) / valid_horizon_denominator
            target_utility = (
                credited_utility
                + float(
                    self.metric_aligned_scorer_cfg.get(
                        "PERMUTATION_TARGET_BOUNDARY_WEIGHT", 0.25
                    )
                )
                * boundary_utility
            ).detach()

            target_temperature = max(
                float(
                    self.metric_aligned_scorer_cfg.get(
                        "PERMUTATION_TARGET_TEMPERATURE", 0.10
                    )
                ),
                self.eps,
            )
            prediction_temperature = max(
                float(
                    self.metric_aligned_scorer_cfg.get(
                        "PERMUTATION_PREDICTION_TEMPERATURE", 0.50
                    )
                ),
                self.eps,
            )
            permutation_target = torch.softmax(
                target_utility / target_temperature, dim=-1
            )
            permutation_log_probability = F.log_softmax(
                permutation_utility / prediction_temperature, dim=-1
            )
            permutation_listwise_sample = -(
                permutation_target * permutation_log_probability
            ).sum(dim=-1)
            loss_permutation_listwise = self._weighted_batch_mean(
                permutation_listwise_sample, sample_weights
            )

            best_target = target_utility.argmax(dim=-1)
            permutation_top1_sample = F.cross_entropy(
                permutation_utility / prediction_temperature,
                best_target,
                reduction="none",
            )
            loss_permutation_top1 = self._weighted_batch_mean(
                permutation_top1_sample, sample_weights
            )

            target_difference = (
                target_utility[:, :, None]
                - target_utility[:, None, :]
            )
            utility_difference = (
                permutation_utility[:, :, None]
                - permutation_utility[:, None, :]
            )
            pair_margin = float(
                self.metric_aligned_scorer_cfg.get(
                    "PERMUTATION_PAIR_TARGET_MARGIN", 0.02
                )
            )
            pair_mask = target_difference.abs() > pair_margin
            pair_loss = F.softplus(
                -target_difference.sign()
                * utility_difference
                / prediction_temperature
            )
            pairwise_sample = (
                pair_loss * pair_mask.type_as(pair_loss)
            ).sum(dim=(1, 2)) / pair_mask.sum(
                dim=(1, 2)
            ).clamp_min(1)
            loss_permutation_pairwise = self._weighted_batch_mean(
                pairwise_sample, sample_weights
            )
            permutation_top1_accuracy = (
                permutation_utility.argmax(dim=-1) == best_target
            ).type_as(permutation_utility).mean()
            sorted_target = target_utility.sort(
                dim=-1, descending=True
            ).values
            permutation_target_margin = (
                sorted_target[:, 0] - sorted_target[:, 1]
            ).mean()

        total = (
            float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_MATCH", 0.5
                )
            )
            * loss_match
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_BOUNDARY", 0.25
                )
            )
            * loss_boundary
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_LISTWISE", 0.5
                )
            )
            * loss_listwise
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_SCENE", 0.1
                )
            )
            * loss_scene
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_GLOBAL_AP", 1.0
                )
            )
            * loss_global_ap
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_NO_MATCH_UNIFORM", 0.2
                )
            )
            * loss_no_match_uniform
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_PRIOR_KL", 0.02
                )
            )
            * loss_prior_kl
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_PERMUTATION_LISTWISE", 0.0
                )
            )
            * loss_permutation_listwise
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_PERMUTATION_PAIRWISE", 0.0
                )
            )
            * loss_permutation_pairwise
            + float(
                self.metric_aligned_scorer_cfg.get(
                    "LOSS_WEIGHT_PERMUTATION_TOP1", 0.0
                )
            )
            * loss_permutation_top1
        )

        with torch.no_grad():
            batch_idx = torch.arange(
                final_probability.shape[0],
                device=final_probability.device,
            )
            final_top = final_probability.argmax(dim=-1)
            base_top = scorer_output["base_logits"].argmax(dim=-1)
            final_match = metrics["horizon_match"][
                batch_idx, final_top
            ].type_as(final_probability)
            base_match = metrics["horizon_match"][
                batch_idx, base_top
            ].type_as(final_probability)
            valid_weight = (
                metrics["pair_valid"].type_as(final_probability)
                * horizon_weights[None]
            )
            valid_denominator = valid_weight.sum().clamp_min(1.0)
            final_top_match = (
                final_match * valid_weight
            ).sum() / valid_denominator
            base_top_match = (
                base_match * valid_weight
            ).sum() / valid_denominator

        return total, {
            "loss_metric_scorer": total,
            "loss_metric_match": loss_match,
            "loss_metric_boundary": loss_boundary,
            "loss_metric_listwise": loss_listwise,
            "loss_metric_scene": loss_scene,
            "loss_metric_global_ap": loss_global_ap,
            "loss_metric_no_match_uniform": loss_no_match_uniform,
            "loss_metric_prior_kl": loss_prior_kl,
            "loss_metric_permutation_listwise": (
                loss_permutation_listwise
            ),
            "loss_metric_permutation_pairwise": (
                loss_permutation_pairwise
            ),
            "loss_metric_permutation_top1": loss_permutation_top1,
            "metric_permutation_top1_accuracy": (
                permutation_top1_accuracy
            ),
            "metric_permutation_target_margin": (
                permutation_target_margin
            ),
            "metric_permutation_delta_abs": scorer_output[
                "permutation_utility_delta"
            ].abs().mean(),
            "metric_final_top1_match": final_top_match,
            "metric_base_top1_match": base_top_match,
            "metric_fusion_gate": scorer_output["fusion_gate"].mean(),
            "metric_score_temperature": scorer_output[
                "score_temperature"
            ].mean(),
            "metric_coverage_probability": scorer_output[
                "coverage_probability"
            ].mean(),
            "metric_coverage_temperature": scorer_output[
                "coverage_temperature"
            ].mean(),
            "metric_base_logit_scale": scorer_output[
                "base_logit_scale"
            ],
            "metric_evidence_abs": scorer_output[
                "evidence_delta"
            ].abs().mean(),
            "joint_oracle_ade": metrics["ade"].min(dim=-1)[0].mean(),
            "joint_oracle_fde": metrics["fde"].min(dim=-1)[0].mean(),
            "metric_positive_probability": (
                (match_probability * target).sum()
                / target.sum().clamp_min(1.0)
            ),
            "metric_negative_probability": (
                (
                    match_probability
                    * (1.0 - target)
                    * supervision
                ).sum()
                / (
                    (1.0 - target) * supervision
                ).sum().clamp_min(1.0)
            ),
        }

    def _deployed_set_permutation_loss(self, final_output, input_dict):
        """Learn final score association without changing values or members."""
        scorer_output = final_output.get("deployed_set_scorer")
        if (
            scorer_output is None
            or "permutation_utility" not in scorer_output
        ):
            raise RuntimeError(
                "Final output is missing deployed-set permutation values"
            )
        trajectories = final_output["deployed_set_trajs"].detach()
        metrics = self._official_match_quality(trajectories, input_dict)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=metrics["horizon_cost"].detach(),
            pair_valid=metrics["pair_valid"],
        )
        cfg = self.deployed_set_scorer_cfg
        utility = scorer_output["permutation_utility"]
        base_logits = scorer_output["pre_permutation_logits"].detach()
        proposal_logits = scorer_output[
            "permutation_proposal_logits"
        ].detach()
        final_logits = scorer_output["joint_logits"]
        sample_weights = self._pair_sample_weights(
            input_dict, utility.device, utility.dtype
        )
        horizon_weights = self.deployed_set_scorer.horizon_weights.type_as(
            utility
        )
        pair_valid = metrics["pair_valid"].type_as(utility)
        valid_horizon_weight = pair_valid * horizon_weights[None]
        horizon_denominator = valid_horizon_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)

        credited_utility = (
            credit["credited_target"].type_as(utility)
            * valid_horizon_weight[:, None]
        ).sum(dim=-1) / horizon_denominator
        boundary_temperature = max(
            float(cfg.get("PERMUTATION_BOUNDARY_TEMPERATURE", 0.20)),
            self.eps,
        )
        boundary_horizon_utility = torch.sigmoid(
            (1.0 - metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        boundary_utility = (
            boundary_horizon_utility
            * valid_horizon_weight[:, None]
        ).sum(dim=-1) / horizon_denominator
        target_utility = (
            credited_utility
            + float(
                cfg.get("PERMUTATION_TARGET_BOUNDARY_WEIGHT", 0.25)
            )
            * boundary_utility
        ).detach()

        target_temperature = max(
            float(cfg.get("PERMUTATION_TARGET_TEMPERATURE", 0.10)),
            self.eps,
        )
        prediction_temperature = max(
            float(
                cfg.get("PERMUTATION_PREDICTION_TEMPERATURE", 0.50)
            ),
            self.eps,
        )
        target_distribution = torch.softmax(
            target_utility / target_temperature, dim=-1
        )
        listwise_sample = -(
            target_distribution
            * F.log_softmax(utility / prediction_temperature, dim=-1)
        ).sum(dim=-1)
        target_span = target_utility.amax(dim=-1) - target_utility.amin(
            dim=-1
        )
        target_margin = float(
            cfg.get("PERMUTATION_PAIR_TARGET_MARGIN", 0.02)
        )
        informative = (
            target_span > target_margin
        ).type_as(utility) * metrics["pair_valid"].any(
            dim=-1
        ).type_as(utility)
        loss_listwise = self._weighted_batch_mean(
            listwise_sample, sample_weights * informative
        )

        target_delta = (
            target_utility[:, :, None] - target_utility[:, None, :]
        )
        predicted_delta = (
            utility[:, :, None] - utility[:, None, :]
        ) / prediction_temperature
        meaningful_pair = target_delta.abs() > target_margin
        num_modes = utility.shape[-1]
        upper_triangle = torch.triu(
            torch.ones(
                num_modes,
                num_modes,
                device=utility.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None]
        pair_mask = meaningful_pair & upper_triangle
        pair_target = (target_delta > 0.0).type_as(utility)
        pair_regret = target_delta.abs().clamp_min(target_margin)
        pair_loss = F.binary_cross_entropy_with_logits(
            predicted_delta, pair_target, reduction="none"
        ) * pair_regret
        pair_weight = pair_mask.type_as(utility)
        pairwise_sample = (pair_loss * pair_weight).sum(dim=(1, 2)) / (
            (pair_regret * pair_weight).sum(dim=(1, 2)).clamp_min(self.eps)
        )
        pairwise_valid = pair_mask.any(dim=2).any(dim=1).type_as(utility)
        loss_pairwise = self._weighted_batch_mean(
            pairwise_sample,
            sample_weights * pairwise_valid,
        )

        target_top = target_utility.argmax(dim=-1)
        top1_sample = F.cross_entropy(
            utility / prediction_temperature,
            target_top,
            reduction="none",
        )
        loss_top1 = self._weighted_batch_mean(
            top1_sample, sample_weights * informative
        )

        base_probability = torch.softmax(base_logits, dim=-1)
        proposal_probability = torch.softmax(proposal_logits, dim=-1)
        base_expected_utility = (
            base_probability * target_utility
        ).sum(dim=-1)
        proposal_expected_utility = (
            proposal_probability * target_utility
        ).sum(dim=-1)
        proposal_gain = (
            proposal_expected_utility - base_expected_utility
        ).detach()
        base_order = base_logits.argsort(dim=-1, descending=True)
        proposal_changed = (
            scorer_output["permutation_order"] != base_order
        ).any(dim=-1)
        acceptance_margin = float(
            cfg.get("PERMUTATION_ACCEPTANCE_GAIN_MARGIN", 0.002)
        )
        acceptance_target = (
            proposal_gain > acceptance_margin
        ).type_as(utility)
        acceptance_logit = scorer_output[
            "permutation_acceptance_logit"
        ]
        acceptance_bce = F.binary_cross_entropy_with_logits(
            acceptance_logit,
            acceptance_target,
            reduction="none",
        )
        false_accept_weight = float(
            cfg.get("PERMUTATION_FALSE_ACCEPT_WEIGHT", 2.0)
        )
        acceptance_bce = acceptance_bce * (
            acceptance_target
            + (1.0 - acceptance_target) * false_accept_weight
        )
        changed_weight = proposal_changed.type_as(utility) * informative
        loss_acceptance = self._weighted_batch_mean(
            acceptance_bce, sample_weights * changed_weight
        )

        gain_scale = max(
            float(cfg.get("PERMUTATION_GAIN_REGRESSION_SCALE", 0.10)),
            self.eps,
        )
        predicted_gain = gain_scale * torch.tanh(acceptance_logit)
        gain_regression_sample = F.smooth_l1_loss(
            predicted_gain,
            proposal_gain.clamp(-gain_scale, gain_scale),
            reduction="none",
            beta=max(gain_scale * 0.1, self.eps),
        )
        loss_gain_regression = self._weighted_batch_mean(
            gain_regression_sample,
            sample_weights * metrics["pair_valid"].any(
                dim=-1
            ).type_as(utility),
        )
        loss_utility_reg = scorer_output[
            "permutation_utility_delta"
        ].square().mean()

        total = (
            float(cfg.get("LOSS_WEIGHT_PERMUTATION_LISTWISE", 1.0))
            * loss_listwise
            + float(cfg.get("LOSS_WEIGHT_PERMUTATION_PAIRWISE", 1.0))
            * loss_pairwise
            + float(cfg.get("LOSS_WEIGHT_PERMUTATION_TOP1", 0.5))
            * loss_top1
            + float(cfg.get("LOSS_WEIGHT_PERMUTATION_ACCEPTANCE", 0.5))
            * loss_acceptance
            + float(
                cfg.get("LOSS_WEIGHT_PERMUTATION_GAIN_REGRESSION", 0.5)
            )
            * loss_gain_regression
            + float(cfg.get("LOSS_WEIGHT_PERMUTATION_REG", 0.001))
            * loss_utility_reg
        )

        with torch.no_grad():
            batch_idx = torch.arange(
                utility.shape[0], device=utility.device
            )
            base_top = base_logits.argmax(dim=-1)
            proposal_top = proposal_logits.argmax(dim=-1)
            final_top = final_logits.argmax(dim=-1)
            accepted = scorer_output["permutation_accepted"]
            correct_acceptance = (
                accepted == acceptance_target.bool()
            ).type_as(utility)
            accepted_changed = accepted & proposal_changed
            accepted_gain = (
                proposal_gain
                * accepted_changed.type_as(utility)
            ).sum() / accepted_changed.sum().clamp_min(1)
            oracle_order = target_utility.argsort(
                dim=-1, descending=True
            )
            sorted_logits = base_logits.sort(
                dim=-1, descending=True
            ).values
            oracle_logits = torch.empty_like(base_logits).scatter(
                1, oracle_order, sorted_logits
            )
            oracle_gain = (
                torch.softmax(oracle_logits, dim=-1) * target_utility
            ).sum(dim=-1) - base_expected_utility
            multiset_error = (
                final_logits.sort(dim=-1).values
                - base_logits.sort(dim=-1).values
            ).abs().amax()

        return total, {
            "loss_final_permutation": total,
            "loss_permutation_listwise": loss_listwise,
            "loss_permutation_pairwise": loss_pairwise,
            "loss_permutation_top1": loss_top1,
            "loss_permutation_acceptance": loss_acceptance,
            "loss_permutation_gain_regression": loss_gain_regression,
            "loss_permutation_utility_reg": loss_utility_reg,
            "permutation_base_top1_utility": self._weighted_batch_mean(
                target_utility[batch_idx, base_top], sample_weights
            ),
            "permutation_proposal_top1_utility": self._weighted_batch_mean(
                target_utility[batch_idx, proposal_top], sample_weights
            ),
            "permutation_final_top1_utility": self._weighted_batch_mean(
                target_utility[batch_idx, final_top], sample_weights
            ),
            "permutation_target_top1_accuracy": (
                proposal_top == target_top
            ).type_as(utility).mean(),
            "permutation_proposal_gain": self._weighted_batch_mean(
                proposal_gain, sample_weights
            ),
            "permutation_oracle_gain": self._weighted_batch_mean(
                oracle_gain, sample_weights
            ),
            "permutation_change_rate": proposal_changed.type_as(
                utility
            ).mean(),
            "permutation_accept_rate": accepted.type_as(utility).mean(),
            "permutation_acceptance_accuracy": correct_acceptance.mean(),
            "permutation_accepted_gain": accepted_gain,
            "permutation_multiset_error": multiset_error,
            "joint_oracle_ade": metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": metrics["fde"].amin(dim=1).mean(),
        }

    def _deployed_set_utility_loss(self, final_output, input_dict):
        scorer_output = final_output.get("deployed_set_scorer")
        if scorer_output is None:
            raise RuntimeError(
                "Final output is missing deployed-set scorer values"
            )
        trajectories = final_output["deployed_set_trajs"].detach()
        metrics = self._official_match_quality(trajectories, input_dict)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=metrics["horizon_cost"].detach(),
            pair_valid=metrics["pair_valid"],
        )
        cfg = self.deployed_set_scorer_cfg
        horizon_logits = scorer_output["horizon_logits"]
        final_logits = scorer_output["joint_logits"]
        final_probability = scorer_output["probability"]
        target = credit["credited_target"].type_as(horizon_logits)
        supervision = credit["supervision_mask"].type_as(horizon_logits)
        pair_valid = metrics["pair_valid"].type_as(horizon_logits)
        sample_weights = self._pair_sample_weights(
            input_dict, horizon_logits.device, horizon_logits.dtype
        )
        horizon_weights = self.deployed_set_scorer.horizon_weights.type_as(
            horizon_logits
        )

        focal_alpha = float(cfg.get("FOCAL_ALPHA", 0.75))
        focal_gamma = float(cfg.get("FOCAL_GAMMA", 2.0))
        match_ce = F.binary_cross_entropy_with_logits(
            horizon_logits, target, reduction="none"
        )
        match_probability = torch.sigmoid(horizon_logits)
        p_t = (
            match_probability * target
            + (1.0 - match_probability) * (1.0 - target)
        )
        alpha_t = (
            focal_alpha * target
            + (1.0 - focal_alpha) * (1.0 - target)
        )
        focal = alpha_t * (1.0 - p_t).pow(focal_gamma) * match_ce
        focal_weight = supervision * horizon_weights[None, None]
        focal_sample = (focal * focal_weight).sum(dim=(1, 2)) / (
            focal_weight.sum(dim=(1, 2)).clamp_min(1.0)
        )
        loss_match = self._weighted_batch_mean(
            focal_sample, sample_weights
        )

        boundary_temperature = max(
            float(cfg.get("BOUNDARY_TEMPERATURE", 0.20)), self.eps
        )
        soft_horizon_utility = torch.sigmoid(
            (1.0 - metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        boundary = F.binary_cross_entropy_with_logits(
            horizon_logits, soft_horizon_utility, reduction="none"
        )
        boundary_weight = (
            pair_valid[:, None].expand_as(boundary)
            * horizon_weights[None, None]
        )
        boundary_sample = (boundary * boundary_weight).sum(dim=(1, 2)) / (
            boundary_weight.sum(dim=(1, 2)).clamp_min(1.0)
        )
        loss_boundary = self._weighted_batch_mean(
            boundary_sample, sample_weights
        )

        list_temperature = max(
            float(cfg.get("LISTWISE_TEMPERATURE", 0.15)), self.eps
        )
        horizon_rank_target = torch.softmax(
            soft_horizon_utility / list_temperature, dim=1
        ).detach()
        horizon_listwise = -(
            horizon_rank_target * F.log_softmax(horizon_logits, dim=1)
        ).sum(dim=1)
        horizon_listwise_sample = (
            horizon_listwise * pair_valid * horizon_weights[None]
        ).sum(dim=-1) / (
            (pair_valid * horizon_weights[None]).sum(dim=-1).clamp_min(1.0)
        )
        loss_listwise = self._weighted_batch_mean(
            horizon_listwise_sample, sample_weights
        )

        valid_horizon_weight = pair_valid * horizon_weights[None]
        aggregate_utility = (
            soft_horizon_utility * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_horizon_weight[:, None].sum(
            dim=-1
        ).clamp_min(self.eps)
        set_temperature = max(
            float(cfg.get("SET_TARGET_TEMPERATURE", 0.10)), self.eps
        )
        set_target = torch.softmax(
            aggregate_utility / set_temperature, dim=-1
        ).detach()
        set_rank_sample = -(
            set_target * F.log_softmax(final_logits, dim=-1)
        ).sum(dim=-1)
        valid_sample = metrics["pair_valid"].any(dim=-1).type_as(final_logits)
        loss_set_rank = self._weighted_batch_mean(
            set_rank_sample, sample_weights * valid_sample
        )

        pair_target_temperature = max(
            float(cfg.get("PAIR_TARGET_TEMPERATURE", 0.08)), self.eps
        )
        pair_score_temperature = max(
            float(cfg.get("PAIR_SCORE_TEMPERATURE", 0.50)), self.eps
        )
        utility_delta = (
            aggregate_utility[:, :, None] - aggregate_utility[:, None, :]
        )
        pair_target = torch.sigmoid(
            utility_delta / pair_target_temperature
        ).detach()
        score_delta = (
            final_logits[:, :, None] - final_logits[:, None, :]
        ) / pair_score_temperature
        pair_loss = F.binary_cross_entropy_with_logits(
            score_delta, pair_target, reduction="none"
        )
        num_modes = final_logits.shape[-1]
        off_diagonal = ~torch.eye(
            num_modes, device=final_logits.device, dtype=torch.bool
        )[None]
        pair_sample = (
            pair_loss * off_diagonal.type_as(pair_loss)
        ).sum(dim=(1, 2)) / off_diagonal.sum().clamp_min(1)
        loss_pairwise = self._weighted_batch_mean(
            pair_sample, sample_weights * valid_sample
        )

        expected_cost_sample = (
            final_probability * (1.0 - aggregate_utility)
        ).sum(dim=-1)
        loss_expected_utility = self._weighted_batch_mean(
            expected_cost_sample, sample_weights * valid_sample
        )

        group_ids = official_ap_group_ids(
            input_dict,
            final_probability.shape[0],
            final_probability.device,
        )
        loss_global_ap = self._global_ap_surrogate(
            final_probability,
            credit,
            group_ids,
            scorer_cfg=cfg,
        )

        no_match_weight = (
            (~credit["has_match"] & metrics["pair_valid"])
            .type_as(final_probability)
            * horizon_weights[None]
        ).sum(dim=-1)
        uniform_kl = (
            final_probability
            * torch.log(
                final_probability.clamp_min(self.eps)
                * final_probability.shape[-1]
            )
        ).sum(dim=-1)
        loss_no_match_uniform = self._weighted_batch_mean(
            uniform_kl, sample_weights * no_match_weight
        )

        base_probability = torch.softmax(
            scorer_output["base_logits"].detach(), dim=-1
        )
        prior_kl = (
            base_probability
            * (
                torch.log(base_probability.clamp_min(self.eps))
                - torch.log(final_probability.clamp_min(self.eps))
            )
        ).sum(dim=-1)
        loss_prior_kl = self._weighted_batch_mean(
            prior_kl, sample_weights
        ).clamp_min(0.0)
        loss_residual_reg = scorer_output["logit_residual"].square().mean()

        total = (
            float(cfg.get("LOSS_WEIGHT_MATCH", 0.50)) * loss_match
            + float(cfg.get("LOSS_WEIGHT_BOUNDARY", 0.25)) * loss_boundary
            + float(cfg.get("LOSS_WEIGHT_LISTWISE", 0.50)) * loss_listwise
            + float(cfg.get("LOSS_WEIGHT_SET_RANK", 1.00)) * loss_set_rank
            + float(cfg.get("LOSS_WEIGHT_PAIRWISE", 0.50)) * loss_pairwise
            + float(cfg.get("LOSS_WEIGHT_GLOBAL_AP", 1.00)) * loss_global_ap
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_UTILITY", 0.25))
            * loss_expected_utility
            + float(cfg.get("LOSS_WEIGHT_NO_MATCH_UNIFORM", 0.10))
            * loss_no_match_uniform
            + float(cfg.get("LOSS_WEIGHT_PRIOR_KL", 0.005)) * loss_prior_kl
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL_REG", 0.01))
            * loss_residual_reg
        )

        with torch.no_grad():
            batch_idx = torch.arange(
                final_logits.shape[0], device=final_logits.device
            )
            final_top = final_logits.argmax(dim=-1)
            base_top = scorer_output["base_logits"].argmax(dim=-1)
            final_top_utility = aggregate_utility[batch_idx, final_top]
            base_top_utility = aggregate_utility[batch_idx, base_top]
            final_top_match = metrics["horizon_match"][
                batch_idx, final_top
            ].type_as(final_logits)
            base_top_match = metrics["horizon_match"][
                batch_idx, base_top
            ].type_as(final_logits)
            match_weight = pair_valid * horizon_weights[None]
            match_denominator = match_weight.sum().clamp_min(1.0)

        return total, {
            "loss_deployed_set_utility": total,
            "loss_deployed_match": loss_match,
            "loss_deployed_boundary": loss_boundary,
            "loss_deployed_listwise": loss_listwise,
            "loss_deployed_set_rank": loss_set_rank,
            "loss_deployed_pairwise": loss_pairwise,
            "loss_deployed_global_ap": loss_global_ap,
            "loss_deployed_expected_utility": loss_expected_utility,
            "loss_deployed_no_match_uniform": loss_no_match_uniform,
            "loss_deployed_prior_kl": loss_prior_kl,
            "loss_deployed_residual_reg": loss_residual_reg,
            "deployed_final_top1_utility": self._weighted_batch_mean(
                final_top_utility, sample_weights * valid_sample
            ),
            "deployed_base_top1_utility": self._weighted_batch_mean(
                base_top_utility, sample_weights * valid_sample
            ),
            "deployed_final_top1_match": (
                final_top_match * match_weight
            ).sum() / match_denominator,
            "deployed_base_top1_match": (
                base_top_match * match_weight
            ).sum() / match_denominator,
            "deployed_top1_change_rate": (final_top != base_top)
            .type_as(final_logits)
            .mean(),
            "deployed_logit_residual_abs": scorer_output[
                "logit_residual"
            ].abs().mean(),
            "deployed_scorer_gate": scorer_output["gate"],
            "joint_oracle_ade": metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": metrics["fde"].amin(dim=1).mean(),
        }

    def _candidate_expansion_loss(
        self, final_output, input_dict, score_only=False
    ):
        expansion_module = (
            self.reciprocal_candidate_expansion
            if final_output.get("reciprocal_augmentation") is not None
            else self.candidate_expansion
        )
        if expansion_module is None:
            raise RuntimeError("Candidate expansion module is unavailable")
        pred = final_output["joint_trajs"]
        logits = final_output["joint_logits"]
        protected = self.num_output_modes
        if pred.shape[1] != protected + self.num_expansion_modes:
            raise RuntimeError(
                "Candidate expansion loss received an unexpected mode count"
            )
        base_pred = pred[:, :protected].detach()
        expansion_pred = pred[:, protected:]
        base_metrics = self._official_match_quality(base_pred, input_dict)
        expansion_metrics = self._official_match_quality(
            expansion_pred, input_dict
        )
        all_metrics = self._official_match_quality(pred, input_dict)
        valid = base_metrics["pair_valid"].type_as(pred)
        horizon_weight = expansion_module.horizon_weights.type_as(
            pred
        )[None]
        sample_weight = self._pair_sample_weights(
            input_dict, pred.device, pred.dtype
        )

        expansion_cost = expansion_metrics["horizon_cost"]
        weighted_denominator = (
            valid * horizon_weight
        ).sum(dim=-1).clamp_min(self.eps)
        weighted_cost = (
            expansion_cost * valid[:, None] * horizon_weight[:, None]
        ).sum(dim=-1) / weighted_denominator[:, None]
        base_weighted_cost = (
            base_metrics["horizon_cost"]
            * valid[:, None]
            * horizon_weight[:, None]
        ).sum(dim=-1) / weighted_denominator[:, None]
        base_best_cost = base_metrics["horizon_cost"].amin(dim=1).detach()
        base_covered = base_metrics["horizon_match"].any(dim=1)

        loss_pair_proposal = logits.new_zeros(())
        loss_pair_proposal_horizon = logits.new_zeros(())
        loss_pair_proposal_rescue = logits.new_zeros(())
        loss_pair_proposal_listwise = logits.new_zeros(())
        loss_pair_proposal_quality = logits.new_zeros(())
        loss_pair_proposal_selection = logits.new_zeros(())
        loss_pair_proposal_geometry = logits.new_zeros(())
        loss_pair_proposal_temporal_reg = logits.new_zeros(())
        pair_proposal_oracle_gain = logits.new_zeros(())
        selected_pair_coverage = logits.new_zeros(
            expansion_module.num_horizons
        )
        if (
            expansion_module.use_learned_pair_selector
            and "candidate_pair_prototype_trajs" in final_output
        ):
            pair_metrics = self._official_match_quality(
                final_output["candidate_pair_prototype_trajs"], input_dict
            )
            pair_match = pair_metrics["horizon_match"].type_as(pred)
            pair_cost_raw = pair_metrics["horizon_cost"]
            pair_cost = pair_cost_raw.detach()
            available = ~final_output["candidate_pair_base_used_mask"]
            proposal_valid = (
                valid[:, None].expand_as(pair_match)
                * available[:, :, None].type_as(pred)
            )
            rescue_target = (
                pair_metrics["horizon_match"]
                & (~base_covered)[:, None]
                & available[:, :, None]
            ).type_as(pred)

            horizon_logits = final_output[
                "candidate_pair_proposal_horizon_logits"
            ]
            horizon_probability = torch.sigmoid(horizon_logits)
            horizon_bce = F.binary_cross_entropy_with_logits(
                horizon_logits, pair_match, reduction="none"
            )
            horizon_pt = (
                horizon_probability * pair_match
                + (1.0 - horizon_probability) * (1.0 - pair_match)
            )
            horizon_alpha = 0.75 * pair_match + 0.25 * (1.0 - pair_match)
            horizon_per_sample = (
                horizon_alpha
                * (1.0 - horizon_pt).square()
                * horizon_bce
                * proposal_valid
            ).sum(dim=(1, 2)) / proposal_valid.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
            loss_pair_proposal_horizon = (
                horizon_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            rescue_logits = final_output[
                "candidate_pair_proposal_rescue_logits"
            ]
            rescue_probability = torch.sigmoid(rescue_logits)
            rescue_bce = F.binary_cross_entropy_with_logits(
                rescue_logits, rescue_target, reduction="none"
            )
            rescue_pt = (
                rescue_probability * rescue_target
                + (1.0 - rescue_probability) * (1.0 - rescue_target)
            )
            rescue_alpha = (
                0.85 * rescue_target + 0.15 * (1.0 - rescue_target)
            )
            rescue_per_sample = (
                rescue_alpha
                * (1.0 - rescue_pt).square()
                * rescue_bce
                * proposal_valid
            ).sum(dim=(1, 2)) / proposal_valid.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
            loss_pair_proposal_rescue = (
                rescue_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            rescue_cost = pair_cost.masked_fill(
                ~rescue_target.bool(), torch.inf
            )
            best_rescue = rescue_cost.argmin(dim=1)
            has_rescue = rescue_target.bool().any(dim=1)
            credited_rescue = torch.zeros_like(rescue_target)
            credited_rescue.scatter_(1, best_rescue[:, None], 1.0)
            credited_rescue *= has_rescue[:, None].type_as(pred)
            selection_logits = final_output[
                "candidate_pair_proposal_selection_logits"
            ].masked_fill(~available, -1e4)
            selection_log_probability = F.log_softmax(
                selection_logits, dim=1
            )
            rescue_listwise = -(
                credited_rescue * selection_log_probability[:, :, None]
            ).sum(dim=1)
            rescue_listwise_weight = has_rescue.type_as(pred) * valid
            rescue_listwise_per_sample = (
                rescue_listwise * rescue_listwise_weight
            ).sum(dim=-1) / rescue_listwise_weight.sum(
                dim=-1
            ).clamp_min(1.0)
            loss_pair_proposal_listwise = (
                rescue_listwise_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            pair_weighted_cost = (
                pair_cost * valid[:, None] * horizon_weight[:, None]
            ).sum(dim=-1) / weighted_denominator[:, None]
            quality_target = torch.softmax(
                -pair_weighted_cost.masked_fill(~available, torch.inf)
                / max(
                    float(
                        self.model_cfg.get(
                            "PAIR_SELECTOR_QUALITY_TEMPERATURE", 0.5
                        )
                    ),
                    self.eps,
                ),
                dim=-1,
            )
            quality_logits = final_output[
                "candidate_pair_proposal_quality_logits"
            ].masked_fill(~available, -1e4)
            quality_per_sample = -(
                quality_target * F.log_softmax(quality_logits, dim=-1)
            ).sum(dim=-1)
            loss_pair_proposal_quality = (
                quality_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            rescue_utility = (
                rescue_target * horizon_weight[:, None]
            ).sum(dim=-1)
            utility_target = torch.softmax(
                (
                    3.0 * rescue_utility
                    - pair_weighted_cost
                    / max(
                        float(
                            self.model_cfg.get(
                                "PAIR_SELECTOR_QUALITY_TEMPERATURE", 0.5
                            )
                        ),
                        self.eps,
                    )
                ).masked_fill(~available, -torch.inf),
                dim=-1,
            )
            selection_per_sample = -(
                utility_target * selection_log_probability
            ).sum(dim=-1)
            loss_pair_proposal_selection = (
                selection_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            geometry_temperature = max(
                float(
                    self.model_cfg.get(
                        "PAIR_SELECTOR_GEOMETRY_TEMPERATURE", 0.25
                    )
                ),
                self.eps,
            )
            pair_geometry_assignment = torch.softmax(
                -pair_cost.masked_fill(
                    ~available[:, :, None], torch.inf
                )
                / geometry_temperature,
                dim=1,
            )
            selected_pair_cost = (
                pair_geometry_assignment * pair_cost_raw
            ).sum(dim=1)
            pair_geometry_weight = (
                valid
                * horizon_weight
                * (
                    1.0
                    + float(
                        self.model_cfg.get(
                            "PAIR_SELECTOR_UNCOVERED_BOOST", 4.0
                        )
                    )
                    * (~base_covered).type_as(pred)
                )
            )
            pair_geometry_per_sample = (
                selected_pair_cost * pair_geometry_weight
            ).sum(dim=-1) / pair_geometry_weight.sum(
                dim=-1
            ).clamp_min(self.eps)
            loss_pair_proposal_geometry = (
                pair_geometry_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)
            loss_pair_proposal_temporal_reg = final_output[
                "candidate_pair_proposal_temporal_delta"
            ].square().mean()

            selected_pair_indices = final_output[
                "candidate_expansion_pair_indices"
            ]
            selected_pair_match = pair_metrics["horizon_match"].gather(
                1,
                selected_pair_indices[:, :, None].expand(
                    -1, -1, expansion_module.num_horizons
                ),
            )
            selected_pair_coverage = (
                selected_pair_match.any(dim=1).type_as(pred) * valid
            ).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)

            loss_pair_proposal = (
                0.25 * loss_pair_proposal_horizon
                + 0.75 * loss_pair_proposal_rescue
                + 1.00 * loss_pair_proposal_listwise
                + 0.25 * loss_pair_proposal_quality
                + 0.50 * loss_pair_proposal_selection
                + float(
                    self.model_cfg.get(
                        "PAIR_SELECTOR_LOSS_WEIGHT_GEOMETRY", 1.0
                    )
                )
                * loss_pair_proposal_geometry
                + float(
                    self.model_cfg.get(
                        "PAIR_SELECTOR_LOSS_WEIGHT_TEMPORAL_REG", 0.01
                    )
                )
                * loss_pair_proposal_temporal_reg
            )
            pair_best_cost = pair_cost.amin(dim=1)
            pair_proposal_oracle_gain = (
                (base_best_cost - pair_best_cost)
                * valid
                * horizon_weight
            ).sum() / (
                valid * horizon_weight
            ).sum().clamp_min(self.eps)
        temperature = max(
            float(
                self.model_cfg.get(
                    "EXPANSION_COVERAGE_TEMPERATURE", 0.25
                )
            ),
            self.eps,
        )
        soft_assignment = torch.softmax(
            -expansion_cost.detach() / temperature, dim=1
        )
        selected_cost = (soft_assignment * expansion_cost).sum(dim=1)
        uncovered = (~base_covered & base_metrics["pair_valid"]).type_as(
            pred
        )
        coverage_weight = (
            valid
            * horizon_weight
            * (
                1.0
                + float(
                    self.model_cfg.get(
                        "EXPANSION_UNCOVERED_BOOST", 4.0
                    )
                )
                * uncovered
            )
        )
        gain_margin = float(
            self.model_cfg.get("EXPANSION_GAIN_MARGIN", 0.05)
        )
        coverage_per_sample = (
            (
                selected_cost
                + F.relu(selected_cost - base_best_cost + gain_margin)
            )
            * coverage_weight
        ).sum(dim=-1) / coverage_weight.sum(dim=-1).clamp_min(self.eps)
        loss_coverage = (
            coverage_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)

        best_expansion = expansion_metrics["quality"].detach().argmin(dim=1)
        batch_index = torch.arange(pred.shape[0], device=pred.device)
        selected_pred = expansion_pred[batch_index, best_expansion]
        regression = F.smooth_l1_loss(
            selected_pred,
            expansion_metrics["gt"],
            reduction="none",
            beta=float(
                self.model_cfg.get("EXPANSION_REGRESSION_BETA", 1.0)
            ),
        ).sum(dim=-1)
        time_weight = torch.linspace(
            0.5,
            1.5,
            self.num_future_frames,
            device=pred.device,
            dtype=pred.dtype,
        )
        regression_weight = (
            expansion_metrics["mask"].type_as(pred)
            * time_weight[None, None]
        )
        regression_per_sample = (
            regression * regression_weight
        ).sum(dim=(1, 2)) / regression_weight.sum(
            dim=(1, 2)
        ).clamp_min(self.eps)
        loss_regression = (
            regression_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)

        loss_reciprocal_conditional = logits.new_zeros(())
        loss_reciprocal_direction_diversity = logits.new_zeros(())
        reciprocal_influencer_ade = logits.new_zeros(())
        reciprocal_reactor_error = logits.new_zeros(())
        reciprocal_parent_reliability = logits.new_zeros(())
        reciprocal_influencer_identity_error = logits.new_zeros(())
        if bool(
            getattr(
                expansion_module,
                "is_reciprocal_conditional",
                False,
            )
        ):
            donor_indices = final_output[
                "candidate_expansion_donor_indices"
            ]
            influencer_indices = final_output[
                "candidate_expansion_influencer_indices"
            ].long()
            reactor_indices = final_output[
                "candidate_expansion_reactor_indices"
            ].long()
            donor_pred = base_pred.gather(
                1,
                donor_indices[..., None, None, None].expand(
                    -1,
                    -1,
                    base_pred.shape[2],
                    base_pred.shape[3],
                    base_pred.shape[4],
                ),
            )
            gt = expansion_metrics["gt"]
            gt_mask = expansion_metrics["mask"]
            gt_expanded = gt[:, None].expand(
                -1, self.num_expansion_modes, -1, -1, -1
            )
            mask_expanded = gt_mask[:, None].expand(
                -1, self.num_expansion_modes, -1, -1
            )
            actor_shape = (
                pred.shape[0],
                self.num_expansion_modes,
                1,
                self.num_future_frames,
            )
            influencer_actor = influencer_indices[
                None, :, None, None
            ].expand(*actor_shape)
            reactor_actor = reactor_indices[
                None, :, None, None
            ].expand(*actor_shape)
            influencer_xy_index = influencer_actor[..., None].expand(
                -1, -1, -1, -1, 2
            )
            reactor_xy_index = reactor_actor[..., None].expand(
                -1, -1, -1, -1, 2
            )

            influencer_pred = donor_pred.gather(
                2, influencer_xy_index
            ).squeeze(2)
            influencer_gt = gt_expanded.gather(
                2, influencer_xy_index
            ).squeeze(2)
            influencer_valid = mask_expanded.gather(
                2, influencer_actor
            ).squeeze(2).type_as(pred)
            influencer_distance = torch.linalg.vector_norm(
                influencer_pred - influencer_gt, dim=-1
            )
            influencer_ade = (
                influencer_distance * influencer_valid
            ).sum(dim=-1) / influencer_valid.sum(dim=-1).clamp_min(1.0)

            reactor_pred = expansion_pred.gather(
                2, reactor_xy_index
            ).squeeze(2)
            reactor_gt = gt_expanded.gather(
                2, reactor_xy_index
            ).squeeze(2)
            reactor_valid = mask_expanded.gather(
                2, reactor_actor
            ).squeeze(2).type_as(pred)
            reactor_point_error = F.smooth_l1_loss(
                reactor_pred,
                reactor_gt,
                reduction="none",
                beta=float(
                    self.model_cfg.get(
                        "RECIPROCAL_REACTOR_REGRESSION_BETA", 0.5
                    )
                ),
            ).sum(dim=-1)
            reactor_error = (
                reactor_point_error * reactor_valid
            ).sum(dim=-1) / reactor_valid.sum(dim=-1).clamp_min(1.0)

            conditional_temperature = max(
                float(
                    self.model_cfg.get(
                        "RECIPROCAL_PARENT_TEMPERATURE", 0.35
                    )
                ),
                self.eps,
            )
            reliability_center = float(
                self.model_cfg.get(
                    "RECIPROCAL_MAX_INFLUENCER_ADE", 2.0
                )
            )
            reliability_scale = max(
                float(
                    self.model_cfg.get(
                        "RECIPROCAL_INFLUENCER_ADE_SCALE", 0.5
                    )
                ),
                self.eps,
            )
            conditional_per_sample = []
            reliability_values = []
            for influencer_agent in (0, 1):
                group = influencer_indices == influencer_agent
                group_influencer_ade = influencer_ade[:, group]
                responsibility = torch.softmax(
                    -group_influencer_ade.detach()
                    / conditional_temperature,
                    dim=-1,
                )
                reliability = torch.sigmoid(
                    (
                        reliability_center
                        - group_influencer_ade.detach()
                    )
                    / reliability_scale
                )
                conditional_weight = responsibility * reliability
                # Keep reliability as an absolute gate.  Renormalizing this
                # product would cancel the signal when every parent poorly
                # explains the influencer and would supervise an unidentifiable
                # conditional response anyway.
                conditional_per_sample.append(
                    (
                        conditional_weight
                        * reactor_error[:, group]
                    ).sum(dim=-1)
                )
                reliability_values.append(reliability.mean(dim=-1))
            conditional_per_sample = torch.stack(
                conditional_per_sample, dim=-1
            ).mean(dim=-1)
            loss_reciprocal_conditional = (
                conditional_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            interaction_probability = final_output[
                "candidate_expansion_interaction_probability"
            ]
            paired_distance = torch.linalg.vector_norm(
                expansion_pred[:, 0::2] - expansion_pred[:, 1::2],
                dim=-1,
            ).mean(dim=(-1, -2))
            paired_risk = interaction_probability.reshape(
                pred.shape[0], 3, 2
            ).mean(dim=-1)
            diversity_margin = float(
                self.model_cfg.get(
                    "RECIPROCAL_DIRECTION_DIVERSITY_MARGIN", 0.75
                )
            )
            reciprocal_diversity_per_sample = F.relu(
                diversity_margin * paired_risk - paired_distance
            ).square().mean(dim=-1)
            loss_reciprocal_direction_diversity = (
                reciprocal_diversity_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

            deployed_influencer = expansion_pred.gather(
                2, influencer_xy_index
            ).squeeze(2)
            reciprocal_influencer_identity_error = (
                deployed_influencer - influencer_pred
            ).abs().amax()
            reciprocal_influencer_ade = influencer_ade.mean()
            reciprocal_reactor_error = reactor_error.mean()
            reciprocal_parent_reliability = torch.stack(
                reliability_values, dim=-1
            ).mean()

        horizon_idx = expansion_module.measurement_steps.to(
            pred.device
        ).clamp_max(self.num_future_frames - 1)
        expansion_endpoint = expansion_pred.index_select(
            3, horizon_idx
        ).permute(0, 3, 1, 2, 4).reshape(
            pred.shape[0],
            expansion_module.num_horizons,
            self.num_expansion_modes,
            -1,
        )
        base_endpoint = base_pred.index_select(
            3, horizon_idx
        ).permute(0, 3, 1, 2, 4).reshape(
            pred.shape[0],
            expansion_module.num_horizons,
            protected,
            -1,
        )
        pairwise_distance = torch.cdist(
            expansion_endpoint, expansion_endpoint
        )
        off_diagonal = ~torch.eye(
            self.num_expansion_modes,
            device=pred.device,
            dtype=torch.bool,
        )[None, None]
        diversity_margin = pred.new_tensor(
            self.model_cfg.get(
                "EXPANSION_DIVERSITY_MARGINS", [1.0, 1.8, 3.0]
            )
        )
        duplicate = F.relu(
            diversity_margin[None, :, None, None] - pairwise_distance
        ).square()
        duplicate = (
            duplicate * off_diagonal.type_as(duplicate)
        ).sum(dim=(2, 3)) / off_diagonal.sum().clamp_min(1)
        nearest_base = torch.cdist(
            expansion_endpoint, base_endpoint
        ).amin(dim=-1)
        novelty_margin = pred.new_tensor(
            self.model_cfg.get(
                "EXPANSION_NOVELTY_MARGINS", [0.5, 0.8, 1.2]
            )
        )
        novelty = F.relu(
            novelty_margin[None, :, None] - nearest_base
        ).square().mean(dim=-1)
        diversity_per_sample = (
            (duplicate + uncovered * novelty) * valid * horizon_weight
        ).sum(dim=-1) / (
            valid * horizon_weight
        ).sum(dim=-1).clamp_min(self.eps)
        loss_diversity = (
            diversity_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)

        improvement = base_best_cost[:, None] - expansion_cost.detach()
        utility_margin = float(
            self.model_cfg.get("EXPANSION_UTILITY_MARGIN", 0.05)
        )
        complementary = (
            expansion_metrics["horizon_match"]
            & ~base_covered[:, None]
            & base_metrics["pair_valid"][:, None]
        )
        improved = (
            (improvement > utility_margin)
            & base_metrics["pair_valid"][:, None]
        )
        replaceable_count = max(
            min(
                self.max_expansion_replacements,
                base_weighted_cost.shape[1],
            ),
            1,
        )
        base_logits = final_output["protected_base_joint_logits"].detach()
        replaceable_index = base_logits.argsort(dim=-1)[
            :, :replaceable_count
        ]
        replaceable_cost = base_weighted_cost.detach().gather(
            1, replaceable_index
        )
        replacement_boundary = replaceable_cost.amax(
            dim=-1, keepdim=True
        )
        replacement_gain = replacement_boundary - weighted_cost.detach()
        admission_logits = final_output[
            "protected_expansion_admission_logits"
        ]
        loss_set_marginal_gain = logits.new_zeros(())
        set_marginal_gain = logits.new_zeros(
            pred.shape[0], self.num_expansion_modes
        )
        set_choice_positive = logits.new_zeros(pred.shape[0])
        loss_scene_replacement_gate = logits.new_zeros(())
        loss_conditional_candidate_selector = logits.new_zeros(())
        loss_conditional_selector_margin = logits.new_zeros(())
        conditional_selector_accuracy = logits.new_zeros(())
        scene_gate_accuracy = logits.new_zeros(())
        if expansion_module.use_set_marginal_gain_target:
            boundary_temperature = max(
                float(
                    self.model_cfg.get(
                        "EXPANSION_SET_BOUNDARY_TEMPERATURE", 0.15
                    )
                ),
                self.eps,
            )
            base_quality = (
                0.8 * base_metrics["horizon_match"].type_as(pred)
                + 0.2
                * torch.sigmoid(
                    (1.0 - base_metrics["horizon_cost"].detach())
                    / boundary_temperature
                )
            )
            expansion_quality = (
                0.8 * expansion_metrics["horizon_match"].type_as(pred)
                + 0.2
                * torch.sigmoid(
                    (1.0 - expansion_cost.detach())
                    / boundary_temperature
                )
            )
            removed_base = base_logits.argmin(dim=-1)
            retained_mask = torch.ones_like(base_logits, dtype=torch.bool)
            retained_mask.scatter_(1, removed_base[:, None], False)
            retained_quality = base_quality.masked_fill(
                ~retained_mask[:, :, None], -torch.inf
            ).amax(dim=1)
            protected_quality = base_quality.amax(dim=1)
            replacement_quality = torch.maximum(
                retained_quality[:, None], expansion_quality
            )
            set_gain_by_horizon = (
                replacement_quality - protected_quality[:, None]
            )
            set_marginal_gain = (
                set_gain_by_horizon
                * valid[:, None]
                * horizon_weight[:, None]
            ).sum(dim=-1) / (
                valid * horizon_weight
            ).sum(dim=-1, keepdim=True).clamp_min(self.eps)
            set_gain_margin = float(
                self.model_cfg.get("EXPANSION_SET_GAIN_MARGIN", 0.01)
            )
            best_set_gain, best_set_candidate = set_marginal_gain.max(
                dim=-1
            )
            set_choice_positive = best_set_gain > set_gain_margin
            if expansion_module.use_hierarchical_set_selector:
                scene_gate_logits = final_output[
                    "protected_expansion_scene_gate_logits"
                ]
                scene_target = set_choice_positive.type_as(scene_gate_logits)
                scene_gate_bce = F.binary_cross_entropy_with_logits(
                    scene_gate_logits, scene_target, reduction="none"
                )
                scene_gate_weight = 1.0 + (
                    float(
                        self.model_cfg.get(
                            "EXPANSION_SCENE_GATE_POS_WEIGHT", 4.0
                        )
                    )
                    - 1.0
                ) * scene_target
                loss_scene_replacement_gate = (
                    scene_gate_bce * scene_gate_weight * sample_weight
                ).sum() / (
                    scene_gate_weight * sample_weight
                ).sum().clamp_min(self.eps)

                selector_logits = final_output[
                    "protected_expansion_candidate_selector_logits"
                ]
                selector_temperature = max(
                    float(
                        self.model_cfg.get(
                            "EXPANSION_CONDITIONAL_SELECTOR_TEMPERATURE", 0.05
                        )
                    ),
                    self.eps,
                )
                soft_selector_target = torch.softmax(
                    set_marginal_gain.detach() / selector_temperature,
                    dim=-1,
                )
                hard_selector_target = torch.zeros_like(selector_logits)
                hard_selector_target.scatter_(
                    1, best_set_candidate[:, None], 1.0
                )
                hard_target_mix = float(
                    self.model_cfg.get(
                        "EXPANSION_CONDITIONAL_HARD_TARGET_MIX", 0.5
                    )
                )
                selector_target = (
                    hard_target_mix * hard_selector_target
                    + (1.0 - hard_target_mix) * soft_selector_target
                )
                selector_per_sample = -(
                    selector_target
                    * F.log_softmax(selector_logits, dim=-1)
                ).sum(dim=-1)
                positive_weight = (
                    set_choice_positive.type_as(sample_weight) * sample_weight
                )
                loss_conditional_candidate_selector = (
                    selector_per_sample * positive_weight
                ).sum() / positive_weight.sum().clamp_min(self.eps)

                best_selector_logit = selector_logits.gather(
                    1, best_set_candidate[:, None]
                ).squeeze(-1)
                hard_negative_logit = selector_logits.masked_fill(
                    hard_selector_target.bool(), -torch.inf
                ).amax(dim=-1)
                selector_margin = float(
                    self.model_cfg.get(
                        "EXPANSION_CONDITIONAL_SELECTOR_MARGIN", 0.25
                    )
                )
                selector_margin_per_sample = F.relu(
                    selector_margin
                    - best_selector_logit
                    + hard_negative_logit
                )
                loss_conditional_selector_margin = (
                    selector_margin_per_sample * positive_weight
                ).sum() / positive_weight.sum().clamp_min(self.eps)
                loss_set_marginal_gain = (
                    float(
                        self.model_cfg.get(
                            "LOSS_WEIGHT_EXPANSION_SCENE_GATE", 1.0
                        )
                    )
                    * loss_scene_replacement_gate
                    + float(
                        self.model_cfg.get(
                            "LOSS_WEIGHT_EXPANSION_CONDITIONAL_SELECTOR", 1.0
                        )
                    )
                    * loss_conditional_candidate_selector
                    + float(
                        self.model_cfg.get(
                            "LOSS_WEIGHT_EXPANSION_CONDITIONAL_MARGIN", 0.25
                        )
                    )
                    * loss_conditional_selector_margin
                )
                with torch.no_grad():
                    positive_count = set_choice_positive.sum().clamp_min(1)
                    conditional_selector_accuracy = (
                        (
                            selector_logits.argmax(dim=-1)
                            == best_set_candidate
                        )
                        & set_choice_positive
                    ).sum().type_as(logits) / positive_count
                    scene_gate_accuracy = (
                        (scene_gate_logits > 0) == set_choice_positive
                    ).type_as(logits).mean()
            else:
                set_choice_target = torch.where(
                    set_choice_positive,
                    best_set_candidate + 1,
                    torch.zeros_like(best_set_candidate),
                )
                no_replacement_logit = admission_logits.new_zeros(
                    admission_logits.shape[0], 1
                )
                set_choice_logits = torch.cat(
                    [no_replacement_logit, admission_logits], dim=-1
                ) / max(
                    float(
                        self.model_cfg.get(
                            "EXPANSION_SET_CHOICE_TEMPERATURE", 1.0
                        )
                    ),
                    self.eps,
                )
                set_choice_per_sample = F.cross_entropy(
                    set_choice_logits,
                    set_choice_target,
                    reduction="none",
                )
                loss_set_marginal_gain = (
                    set_choice_per_sample * sample_weight
                ).sum() / sample_weight.sum().clamp_min(self.eps)

            admission_target = torch.zeros_like(admission_logits)
            admission_target.scatter_(
                1,
                best_set_candidate[:, None],
                set_choice_positive[:, None].type_as(admission_target),
            )
            horizon_target = (
                set_gain_by_horizon
                > float(
                    self.model_cfg.get(
                        "EXPANSION_SET_HORIZON_GAIN_MARGIN", 0.0
                    )
                )
            ).type_as(pred)
        elif (
            expansion_module.use_decoupled_confidence
            or expansion_module.competitive_replacement_scoring
        ):
            horizon_target = (
                expansion_metrics["horizon_match"]
                & base_metrics["pair_valid"][:, None]
            ).type_as(pred)
            admission_target = (
                (
                    weighted_cost.detach()
                    < replacement_boundary - utility_margin
                )
                | complementary.any(dim=-1)
            ).type_as(pred)
        else:
            horizon_target = (complementary | improved).type_as(pred)
            admission_target = horizon_target.any(dim=-1).type_as(pred)
        if expansion_module.use_hierarchical_set_selector:
            loss_admission = logits.new_zeros(())
        else:
            admission_bce = F.binary_cross_entropy_with_logits(
                admission_logits, admission_target, reduction="none"
            )
            admission_weight = 1.0 + (
                float(
                    self.model_cfg.get(
                        "EXPANSION_ADMISSION_POS_WEIGHT", 4.0
                    )
                )
                - 1.0
            ) * admission_target
            admission_per_sample = (
                admission_bce * admission_weight
            ).mean(dim=-1)
            loss_admission = (
                admission_per_sample * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

        loss_replacement_utility = logits.new_zeros(())
        if expansion_module.competitive_replacement_scoring:
            target_scale = max(
                float(
                    self.model_cfg.get(
                        "EXPANSION_REPLACEMENT_TARGET_SCALE", 1.0
                    )
                ),
                self.eps,
            )
            complementary_any = complementary.any(dim=-1)
            replacement_target = torch.where(
                complementary_any,
                torch.maximum(
                    replacement_gain,
                    replacement_gain.new_full(
                        replacement_gain.shape, utility_margin
                    ),
                ),
                replacement_gain,
            )
            replacement_target = torch.tanh(
                replacement_target / target_scale
            )
            replacement_prediction = torch.tanh(
                final_output[
                    "protected_expansion_replacement_utility"
                ]
            )
            replacement_error = F.smooth_l1_loss(
                replacement_prediction,
                replacement_target,
                reduction="none",
                beta=float(
                    self.model_cfg.get(
                        "EXPANSION_REPLACEMENT_REGRESSION_BETA", 0.25
                    )
                ),
            ).mean(dim=-1)
            loss_replacement_utility = (
                replacement_error * sample_weight
            ).sum() / sample_weight.sum().clamp_min(self.eps)

        horizon_logits = final_output[
            "candidate_expansion_horizon_logits"
        ]
        horizon_bce = F.binary_cross_entropy_with_logits(
            horizon_logits, horizon_target, reduction="none"
        )
        horizon_bce = horizon_bce * (
            1.0
            + (
                float(
                    self.model_cfg.get(
                        "EXPANSION_HORIZON_POS_WEIGHT", 1.0
                    )
                )
                - 1.0
            )
            * horizon_target
        )
        horizon_valid = base_metrics["pair_valid"][
            :, None
        ].expand_as(horizon_bce).type_as(horizon_bce)
        horizon_per_sample = (
            horizon_bce * horizon_valid
        ).sum(dim=(1, 2)) / horizon_valid.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_horizon = (
            horizon_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)

        expansion_logits = logits[:, protected:]
        rank_target = torch.softmax(
            -weighted_cost.detach()
            / max(
                float(
                    self.model_cfg.get(
                        "EXPANSION_RANK_TEMPERATURE", 0.5
                    )
                ),
                self.eps,
            ),
            dim=-1,
        )
        loss_rank = -(
            rank_target * F.log_softmax(expansion_logits, dim=-1)
        ).sum(dim=-1)
        loss_rank = (
            loss_rank * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)

        deployment_aligned = bool(
            self.model_cfg.get(
                "EXPANSION_DEPLOYMENT_ALIGNED_CONFIDENCE", False
            )
        )
        deployment_indices = None
        selected_expansion_mask = None
        selected_base_mask = None
        if deployment_aligned:
            deployment_indices = self._guarded_expansion_indices(
                final_output
            )
            selected_expansion_count = torch.zeros(
                pred.shape[0],
                self.num_expansion_modes,
                device=pred.device,
                dtype=torch.long,
            )
            local_expansion_index = deployment_indices - protected
            valid_expansion_index = local_expansion_index >= 0
            selected_expansion_count.scatter_add_(
                1,
                local_expansion_index.clamp(
                    0, self.num_expansion_modes - 1
                ),
                valid_expansion_index.long(),
            )
            selected_expansion_mask = selected_expansion_count > 0
            selected_base_count = torch.zeros(
                pred.shape[0],
                protected,
                device=pred.device,
                dtype=torch.long,
            )
            valid_base_index = deployment_indices < protected
            selected_base_count.scatter_add_(
                1,
                deployment_indices.clamp(0, protected - 1),
                valid_base_index.long(),
            )
            selected_base_mask = selected_base_count > 0

        if expansion_module.use_decoupled_confidence:
            cross_temperature = max(
                float(
                    self.model_cfg.get(
                        "EXPANSION_CROSS_RANK_TEMPERATURE", 0.5
                    )
                ),
                self.eps,
            )
            cross_target = torch.sigmoid(
                (
                    base_weighted_cost.detach()[:, None]
                    - weighted_cost.detach()[:, :, None]
                )
                / cross_temperature
            )
            cross_logit = (
                expansion_logits[:, :, None]
                - logits[:, None, :protected].detach()
            )
            cross_bce = F.binary_cross_entropy_with_logits(
                cross_logit, cross_target, reduction="none"
            )
            cross_weight = 0.5 + 2.0 * (
                cross_target - 0.5
            ).abs()
            if deployment_aligned:
                deploy_pair = (
                    selected_expansion_mask[:, :, None]
                    & selected_base_mask[:, None]
                ).type_as(cross_weight)
                weighted_cross = cross_bce * cross_weight * deploy_pair
                cross_per_sample = weighted_cross.sum(dim=(1, 2)) / (
                    cross_weight * deploy_pair
                ).sum(dim=(1, 2)).clamp_min(1.0)
                cross_scene_weight = selected_expansion_mask.any(
                    dim=-1
                ).type_as(sample_weight)
            else:
                cross_per_sample = (
                    cross_bce * cross_weight
                ).mean(dim=(1, 2))
                cross_scene_weight = torch.ones_like(sample_weight)
            loss_cross_rank = (
                cross_per_sample * sample_weight * cross_scene_weight
            ).sum() / (
                sample_weight * cross_scene_weight
            ).sum().clamp_min(self.eps)
        else:
            loss_cross_rank = logits.new_zeros(())

        credit_logits = logits
        credit_horizon_match = all_metrics["horizon_match"]
        credit_horizon_cost = all_metrics["horizon_cost"].detach()
        if deployment_aligned:
            gather_horizon = deployment_indices[..., None].expand(
                -1, -1, credit_horizon_match.shape[-1]
            )
            credit_logits = logits.gather(1, deployment_indices)
            credit_horizon_match = credit_horizon_match.gather(
                1, gather_horizon
            )
            credit_horizon_cost = credit_horizon_cost.gather(
                1, gather_horizon
            )
        credit = build_soft_map_credit_targets(
            horizon_match=credit_horizon_match,
            horizon_cost=credit_horizon_cost,
            pair_valid=all_metrics["pair_valid"],
        )
        log_probability = F.log_softmax(credit_logits, dim=-1)
        target = credit["credited_target"].type_as(log_probability)
        credit_per_horizon = -(
            target * log_probability[:, :, None]
        ).sum(dim=1)
        credit_weight = (
            credit["has_match"].type_as(log_probability)
            * all_metrics["pair_valid"].type_as(log_probability)
            * horizon_weight
        )
        credit_per_sample = (
            credit_per_horizon * credit_weight
        ).sum(dim=-1) / credit_weight.sum(dim=-1).clamp_min(self.eps)
        loss_credit = (
            credit_per_sample * sample_weight
        ).sum() / sample_weight.sum().clamp_min(self.eps)
        loss_expansion_global_ap = logits.new_zeros(())
        if expansion_module.use_unified_deployed_confidence:
            group_ids = official_ap_group_ids(
                input_dict,
                credit_logits.shape[0],
                credit_logits.device,
            )
            loss_expansion_global_ap = self._global_ap_surrogate(
                torch.softmax(credit_logits, dim=-1),
                credit,
                group_ids,
                scorer_cfg={
                    "GLOBAL_AP_TEMPERATURE": float(
                        self.model_cfg.get(
                            "EXPANSION_GLOBAL_AP_TEMPERATURE", 0.05
                        )
                    )
                },
            )

        loss_score_reg = final_output[
            "candidate_expansion_score_delta"
        ].square().mean()
        unified_confidence_residual = final_output.get(
            "candidate_expansion_unified_confidence_residual"
        )
        unified_confidence_temperature = final_output.get(
            "candidate_expansion_unified_confidence_temperature"
        )
        if expansion_module.use_unified_deployed_confidence:
            loss_unified_confidence_reg = (
                unified_confidence_residual.square().mean()
                + torch.log(
                    unified_confidence_temperature.clamp_min(self.eps)
                ).square().mean()
            )
        else:
            loss_unified_confidence_reg = logits.new_zeros(())
        loss_waypoint_reg = final_output[
            "candidate_expansion_waypoint_delta"
        ].square().mean()
        dense_knot_delta = final_output.get(
            "candidate_expansion_dense_knot_delta"
        )
        dense_temporal_delta = final_output.get(
            "candidate_expansion_dense_temporal_delta"
        )
        if dense_knot_delta is not None and dense_knot_delta.numel() > 0:
            loss_dense_knot_reg = dense_knot_delta.square().mean()
            dense_acceleration = (
                dense_temporal_delta[..., 2:, :]
                - 2.0 * dense_temporal_delta[..., 1:-1, :]
                + dense_temporal_delta[..., :-2, :]
            )
            loss_dense_temporal_smoothness = (
                dense_acceleration.square().mean()
            )
        else:
            loss_dense_knot_reg = logits.new_zeros(())
            loss_dense_temporal_smoothness = logits.new_zeros(())
        total = (
            float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_COVERAGE", 1.0
                )
            )
            * loss_coverage
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_REGRESSION", 0.5
                )
            )
            * loss_regression
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_RECIPROCAL_CONDITIONAL", 0.0
                )
            )
            * loss_reciprocal_conditional
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_RECIPROCAL_DIRECTION_DIVERSITY", 0.0
                )
            )
            * loss_reciprocal_direction_diversity
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_DIVERSITY", 0.02
                )
            )
            * loss_diversity
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_ADMISSION", 0.75
                )
            )
            * loss_admission
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_HORIZON", 0.5
                )
            )
            * loss_horizon
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_RANK", 0.5
                )
            )
            * loss_rank
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_CROSS_RANK", 0.0
                )
            )
            * loss_cross_rank
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_CREDIT", 0.5
                )
            )
            * loss_credit
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_GLOBAL_AP", 0.0
                )
            )
            * loss_expansion_global_ap
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_SCORE_REG", 0.01
                )
            )
            * loss_score_reg
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_WAYPOINT_REG", 0.005
                )
            )
            * loss_waypoint_reg
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_DENSE_KNOT_REG", 0.002
                )
            )
            * loss_dense_knot_reg
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_DENSE_SMOOTHNESS", 0.02
                )
            )
            * loss_dense_temporal_smoothness
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_SET_GAIN", 0.0
                )
            )
            * loss_set_marginal_gain
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_PAIR_PROPOSAL", 1.0
                )
            )
            * loss_pair_proposal
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_EXPANSION_REPLACEMENT_UTILITY", 0.0
                )
            )
            * loss_replacement_utility
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_UNIFIED_CONFIDENCE_REG", 0.0
                )
            )
            * loss_unified_confidence_reg
        )
        if score_only:
            # Phase B activates only the already deployed confidence,
            # admission and retention objectives. Geometry terms are restored
            # unchanged when candidate co-adaptation begins in Phase C.
            total = (
                float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_ADMISSION", 0.75
                    )
                )
                * loss_admission
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_HORIZON", 0.5
                    )
                )
                * loss_horizon
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_RANK", 0.5
                    )
                )
                * loss_rank
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_CROSS_RANK", 0.0
                    )
                )
                * loss_cross_rank
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_CREDIT", 0.5
                    )
                )
                * loss_credit
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_GLOBAL_AP", 0.0
                    )
                )
                * loss_expansion_global_ap
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_SCORE_REG", 0.01
                    )
                )
                * loss_score_reg
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_SET_GAIN", 0.0
                    )
                )
                * loss_set_marginal_gain
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_EXPANSION_REPLACEMENT_UTILITY", 0.0
                    )
                )
                * loss_replacement_utility
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_UNIFIED_CONFIDENCE_REG", 0.0
                    )
                )
                * loss_unified_confidence_reg
            )

        all_best_cost = all_metrics["horizon_cost"].amin(dim=1)
        valid_denominator = valid.sum(dim=0).clamp_min(1.0)
        base_coverage = (
            base_covered.type_as(valid) * valid
        ).sum(dim=0) / valid_denominator
        all_coverage = (
            all_metrics["horizon_match"].any(dim=1).type_as(valid) * valid
        ).sum(dim=0) / valid_denominator
        oracle_gain = (
            (base_best_cost - all_best_cost) * valid * horizon_weight
        ).sum() / (
            valid * horizon_weight
        ).sum().clamp_min(self.eps)
        reciprocal_replacement = final_output.get(
            "reciprocal_set_replacement"
        )
        if reciprocal_replacement is not None:
            deployed_rate = reciprocal_replacement["accepted"].type_as(
                pred
            ).mean()
        elif selected_expansion_mask is not None:
            deployed_rate = selected_expansion_mask.type_as(pred).mean()
        else:
            deployed_rate = pred.new_zeros(())
        return total, {
            "loss_candidate_expansion": total,
            "loss_expansion_coverage": loss_coverage,
            "loss_expansion_regression": loss_regression,
            "loss_reciprocal_conditional": (
                loss_reciprocal_conditional
            ),
            "loss_reciprocal_direction_diversity": (
                loss_reciprocal_direction_diversity
            ),
            "reciprocal_influencer_ade": reciprocal_influencer_ade,
            "reciprocal_reactor_error": reciprocal_reactor_error,
            "reciprocal_parent_reliability": (
                reciprocal_parent_reliability
            ),
            "reciprocal_influencer_identity_error": (
                reciprocal_influencer_identity_error
            ),
            "loss_expansion_diversity": loss_diversity,
            "loss_expansion_admission": loss_admission,
            "loss_expansion_replacement_utility": (
                loss_replacement_utility
            ),
            "loss_expansion_horizon": loss_horizon,
            "loss_expansion_rank": loss_rank,
            "loss_expansion_cross_rank": loss_cross_rank,
            "loss_expansion_credit": loss_credit,
            "loss_expansion_global_ap": loss_expansion_global_ap,
            "loss_expansion_scene_gate": loss_scene_replacement_gate,
            "loss_expansion_conditional_selector": (
                loss_conditional_candidate_selector
            ),
            "loss_expansion_conditional_margin": (
                loss_conditional_selector_margin
            ),
            "loss_unified_confidence_reg": loss_unified_confidence_reg,
            "loss_expansion_dense_knot_reg": loss_dense_knot_reg,
            "loss_expansion_dense_smoothness": (
                loss_dense_temporal_smoothness
            ),
            "loss_expansion_set_gain": loss_set_marginal_gain,
            "loss_pair_proposal": loss_pair_proposal,
            "loss_pair_proposal_horizon": loss_pair_proposal_horizon,
            "loss_pair_proposal_rescue": loss_pair_proposal_rescue,
            "loss_pair_proposal_listwise": loss_pair_proposal_listwise,
            "loss_pair_proposal_quality": loss_pair_proposal_quality,
            "loss_pair_proposal_selection": loss_pair_proposal_selection,
            "loss_pair_proposal_geometry": loss_pair_proposal_geometry,
            "loss_pair_proposal_temporal_reg": (
                loss_pair_proposal_temporal_reg
            ),
            "pair_proposal_oracle_gain": pair_proposal_oracle_gain,
            "pair_proposal_gate": final_output.get(
                "candidate_pair_proposal_gate", logits.new_zeros(())
            ),
            "pair_proposal_selected_coverage_3s": selected_pair_coverage[0],
            "pair_proposal_selected_coverage_5s": selected_pair_coverage[1],
            "pair_proposal_selected_coverage_8s": selected_pair_coverage[2],
            "candidate_expansion_oracle_gain": oracle_gain,
            "candidate_expansion_base_coverage_3s": base_coverage[0],
            "candidate_expansion_base_coverage_5s": base_coverage[1],
            "candidate_expansion_base_coverage_8s": base_coverage[2],
            "candidate_expansion_all_coverage_3s": all_coverage[0],
            "candidate_expansion_all_coverage_5s": all_coverage[1],
            "candidate_expansion_all_coverage_8s": all_coverage[2],
            "candidate_expansion_admission_target": admission_target.mean(),
            "candidate_expansion_admission_probability": torch.sigmoid(
                admission_logits
            ).mean(),
            "candidate_expansion_replacement_utility": final_output.get(
                "protected_expansion_replacement_utility",
                logits.new_zeros(1),
            ).mean(),
            "candidate_expansion_replacement_target_rate": (
                admission_target.mean()
            ),
            "candidate_expansion_set_gain": set_marginal_gain.max(
                dim=-1
            ).values.mean(),
            "candidate_expansion_set_choice_positive_rate": (
                set_choice_positive.type_as(pred).mean()
            ),
            "candidate_expansion_scene_gate_probability": torch.sigmoid(
                final_output[
                    "protected_expansion_scene_gate_logits"
                ]
            ).mean(),
            "candidate_expansion_scene_gate_accuracy": scene_gate_accuracy,
            "candidate_expansion_conditional_selector_accuracy": (
                conditional_selector_accuracy
            ),
            "candidate_expansion_goal_gate": final_output[
                "candidate_expansion_goal_gate"
            ].mean(),
            "candidate_expansion_confidence_gate": final_output[
                "candidate_expansion_confidence_gate"
            ],
            "candidate_expansion_confidence_temperature": (
                final_output[
                    "candidate_expansion_unified_confidence_temperature"
                ].mean()
            ),
            "candidate_expansion_confidence_residual_abs": (
                final_output[
                    "candidate_expansion_unified_confidence_residual"
                ].abs().mean()
            ),
            "candidate_expansion_deployed_rate": deployed_rate,
            "joint_oracle_ade": all_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": all_metrics["fde"].amin(dim=1).mean(),
        }

    def _end_to_end_deployed_set_loss(self, final_output, input_dict):
        """Optimize the exact six-mode set consumed by Waymo evaluation.

        Candidate-bank geometry, admission, and confidence have their own
        auxiliary objectives.  Those objectives do not, by themselves,
        guarantee that the hard 12-to-6 deployment decision is useful as one
        set.  This loss gathers that exact set and jointly supervises its
        confidence ordering and trajectory coverage at the official horizons.
        """
        selected_indices = self._guarded_expansion_indices(final_output)
        batch_size = selected_indices.shape[0]
        batch_idx = torch.arange(
            batch_size, device=selected_indices.device
        )[:, None]
        trajectories = final_output["joint_trajs"][
            batch_idx, selected_indices
        ]
        logits = final_output["joint_logits"][batch_idx, selected_indices]
        metrics = self._official_match_quality(trajectories, input_dict)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=metrics["horizon_cost"].detach(),
            pair_valid=metrics["pair_valid"],
        )
        cfg = self.model_cfg.get("END_TO_END_DEPLOYED_SET", {})
        sample_weight = self._pair_sample_weights(
            input_dict, logits.device, logits.dtype
        )
        horizon_weight = logits.new_tensor(
            cfg.get("HORIZON_WEIGHTS", [0.20, 0.30, 0.50])
        )
        horizon_weight = horizon_weight / horizon_weight.sum().clamp_min(
            self.eps
        )
        valid_horizon = metrics["pair_valid"].type_as(logits)

        # A credited match is the desired ranking target.  When no candidate
        # passes the official threshold, retain a dense geometric target so
        # the confidence branch still receives a useful learning signal.
        fallback_temperature = max(
            float(cfg.get("FALLBACK_TEMPERATURE", 0.50)), self.eps
        )
        fallback_target = torch.softmax(
            -metrics["horizon_cost"].detach() / fallback_temperature,
            dim=1,
        )
        credited_target = credit["credited_target"].type_as(logits)
        target = torch.where(
            credit["has_match"][:, None],
            credited_target,
            fallback_target,
        )
        horizon_ce = -(
            target * F.log_softmax(logits, dim=1)[..., None]
        ).sum(dim=1)
        confidence_per_sample = (
            horizon_ce * valid_horizon * horizon_weight[None]
        ).sum(dim=-1) / (
            valid_horizon * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_confidence = self._weighted_batch_mean(
            confidence_per_sample, sample_weight
        )

        group_ids = official_ap_group_ids(
            input_dict, batch_size, logits.device
        )
        loss_global_ap = self._global_ap_surrogate(
            torch.softmax(logits, dim=-1),
            credit,
            group_ids,
            scorer_cfg={
                "GLOBAL_AP_TEMPERATURE": float(
                    cfg.get("GLOBAL_AP_TEMPERATURE", 0.05)
                )
            },
        )

        # A detached soft assignment avoids a hard winner discontinuity while
        # sending geometric gradients into every near-best deployed mode.
        geometry_temperature = max(
            float(cfg.get("GEOMETRY_TEMPERATURE", 0.35)), self.eps
        )
        geometry_target = torch.softmax(
            -metrics["horizon_cost"].detach() / geometry_temperature,
            dim=1,
        )
        expected_horizon_cost = (
            geometry_target * metrics["horizon_cost"]
        ).sum(dim=1)
        coverage_per_sample = (
            expected_horizon_cost * valid_horizon * horizon_weight[None]
        ).sum(dim=-1) / (
            valid_horizon * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        best_idx = metrics["quality"].detach().argmin(dim=-1)
        best_trajectory = trajectories[
            torch.arange(batch_size, device=logits.device), best_idx
        ]
        regression = F.smooth_l1_loss(
            best_trajectory,
            metrics["gt"],
            reduction="none",
            beta=float(cfg.get("REGRESSION_BETA", 1.0)),
        ).sum(dim=-1)
        regression_per_sample = (
            regression * metrics["mask"]
        ).sum(dim=(-1, -2)) / metrics["mask"].sum(
            dim=(-1, -2)
        ).clamp_min(1)
        loss_regression = self._weighted_batch_mean(
            regression_per_sample, sample_weight
        )

        total = (
            float(cfg.get("LOSS_WEIGHT_CONFIDENCE", 1.0))
            * loss_confidence
            + float(cfg.get("LOSS_WEIGHT_GLOBAL_AP", 0.5))
            * loss_global_ap
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 0.5))
            * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_REGRESSION", 0.25))
            * loss_regression
        )
        selected_expansion_rate = (
            selected_indices >= self.num_output_modes
        ).type_as(logits).mean()
        return total, {
            "loss_end_to_end_deployed_set": total,
            "loss_end_to_end_deployed_confidence": loss_confidence,
            "loss_end_to_end_deployed_global_ap": loss_global_ap,
            "loss_end_to_end_deployed_coverage": loss_coverage,
            "loss_end_to_end_deployed_regression": loss_regression,
            "end_to_end_deployed_expansion_rate": (
                selected_expansion_rate
            ),
            "end_to_end_deployed_credited_rate": (
                credit["has_match"].type_as(logits) * valid_horizon
            ).sum() / valid_horizon.sum().clamp_min(1.0),
            "end_to_end_deployed_oracle_ade": metrics["ade"].amin(
                dim=1
            ).mean(),
            "end_to_end_deployed_oracle_fde": metrics["fde"].amin(
                dim=1
            ).mean(),
        }

    def _deployed_trajectory_refiner_loss(
        self, final_output, input_dict
    ):
        refiner_output = final_output.get("deployed_refiner_output")
        if refiner_output is None:
            raise RuntimeError(
                "Missing deployed-refiner output during geometry training"
            )
        refined_trajs = refiner_output["trajectories"]
        base_trajs = refiner_output["base_trajectories"].detach()
        logits = final_output["deployed_refiner_logits"].detach()
        cfg = self.deployed_trajectory_refiner_cfg

        refined_metrics = self._official_match_quality(
            refined_trajs, input_dict
        )
        with torch.no_grad():
            base_metrics = self._official_match_quality(
                base_trajs, input_dict
            )
        valid = refined_metrics["pair_valid"].type_as(refined_trajs)
        horizon_weight = (
            self.deployed_trajectory_refiner.horizon_weights.type_as(
                refined_trajs
            )[None]
        )
        sample_weight = self._pair_sample_weights(
            input_dict, refined_trajs.device, refined_trajs.dtype
        )
        valid_horizon_weight = valid * horizon_weight
        valid_denominator = valid_horizon_weight.sum(
            dim=-1
        ).clamp_min(self.eps)

        score_temperature = max(
            float(cfg.get("SCORE_TEMPERATURE", 1.0)), self.eps
        )
        probability = torch.softmax(
            logits / score_temperature, dim=-1
        )
        refined_cost = refined_metrics["horizon_cost"]
        base_cost = base_metrics["horizon_cost"].detach()
        target_cost = float(cfg.get("TARGET_NORMALIZED_COST", 0.90))
        threshold_temperature = max(
            float(cfg.get("THRESHOLD_TEMPERATURE", 0.12)), self.eps
        )
        threshold_penalty = threshold_temperature * F.softplus(
            (refined_cost - target_cost) / threshold_temperature
        )
        score_aligned_per_sample = (
            threshold_penalty
            * probability[:, :, None]
            * valid_horizon_weight[:, None]
        ).sum(dim=(1, 2)) / valid_denominator
        loss_score_aligned = self._weighted_batch_mean(
            score_aligned_per_sample, sample_weight
        )

        coverage_temperature = max(
            float(cfg.get("COVERAGE_TEMPERATURE", 0.20)), self.eps
        )
        num_modes = refined_cost.shape[1]
        softmin_cost = -coverage_temperature * (
            torch.logsumexp(
                -refined_cost / coverage_temperature, dim=1
            )
            - math.log(max(num_modes, 1))
        )
        coverage_penalty = threshold_temperature * F.softplus(
            (softmin_cost - target_cost) / threshold_temperature
        )
        coverage_per_sample = (
            coverage_penalty * valid_horizon_weight
        ).sum(dim=-1) / valid_denominator
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        responsibility_temperature = max(
            float(cfg.get("RESPONSIBILITY_TEMPERATURE", 0.50)), self.eps
        )
        base_weighted_cost = (
            base_cost * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_denominator[:, None]
        normalized_logits = (
            logits - logits.mean(dim=-1, keepdim=True)
        ) / logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        responsibility = torch.softmax(
            -base_weighted_cost / responsibility_temperature
            + float(cfg.get("RESPONSIBILITY_SCORE_WEIGHT", 0.25))
            * normalized_logits,
            dim=-1,
        )
        ade_per_sample = (
            responsibility * refined_metrics["ade"]
        ).sum(dim=-1)
        loss_ade = self._weighted_batch_mean(
            ade_per_sample, sample_weight
        )

        allowed_regression = float(
            cfg.get("NO_HARM_COST_TOLERANCE", 0.03)
        )
        no_harm_penalty = F.relu(
            refined_cost - base_cost - allowed_regression
        )
        no_harm_per_sample = (
            no_harm_penalty
            * probability[:, :, None]
            * valid_horizon_weight[:, None]
        ).sum(dim=(1, 2)) / valid_denominator
        loss_no_harm = self._weighted_batch_mean(
            no_harm_per_sample, sample_weight
        )

        steps = self.deployed_trajectory_refiner.measurement_steps.to(
            refined_trajs.device
        )
        base_signature = base_trajs.index_select(3, steps).reshape(
            base_trajs.shape[0], base_trajs.shape[1], -1
        )
        refined_signature = refined_trajs.index_select(3, steps).reshape(
            refined_trajs.shape[0], refined_trajs.shape[1], -1
        )
        base_distance = torch.cdist(
            base_signature.float(), base_signature.float()
        ).type_as(refined_trajs)
        refined_distance = torch.cdist(
            refined_signature.float(), refined_signature.float()
        ).type_as(refined_trajs)
        pair_mask = torch.triu(
            torch.ones(
                num_modes,
                num_modes,
                device=refined_trajs.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None]
        pair_mask = pair_mask & (
            base_distance
            > float(cfg.get("DIVERSITY_MIN_BASE_DISTANCE", 0.5))
        )
        diversity_penalty = F.relu(
            float(cfg.get("DIVERSITY_PRESERVE_RATIO", 0.85))
            * base_distance
            - refined_distance
        )
        diversity_per_sample = (
            diversity_penalty * pair_mask.type_as(diversity_penalty)
        ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)
        loss_diversity = self._weighted_batch_mean(
            diversity_per_sample, sample_weight
        )

        dense_delta = refiner_output["dense_delta"]
        loss_delta_reg = dense_delta.square().mean()
        delta_velocity = dense_delta[:, :, :, 1:] - dense_delta[:, :, :, :-1]
        delta_acceleration = (
            delta_velocity[:, :, :, 1:]
            - delta_velocity[:, :, :, :-1]
        )
        loss_delta_smooth = delta_acceleration.square().mean()

        total = (
            float(cfg.get("LOSS_WEIGHT_SCORE_ALIGNED", 1.0))
            * loss_score_aligned
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 1.0))
            * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_ADE", 0.10)) * loss_ade
            + float(cfg.get("LOSS_WEIGHT_NO_HARM", 1.0))
            * loss_no_harm
            + float(cfg.get("LOSS_WEIGHT_DIVERSITY", 0.05))
            * loss_diversity
            + float(cfg.get("LOSS_WEIGHT_DELTA_REG", 0.01))
            * loss_delta_reg
            + float(cfg.get("LOSS_WEIGHT_DELTA_SMOOTH", 0.02))
            * loss_delta_smooth
        )

        score_weight = (
            probability[:, :, None] * valid_horizon_weight[:, None]
        )
        base_score_cost = (
            base_cost * score_weight
        ).sum() / score_weight.sum().clamp_min(self.eps)
        refined_score_cost = (
            refined_cost * score_weight
        ).sum() / score_weight.sum().clamp_min(self.eps)
        base_match = base_metrics["horizon_match"].type_as(refined_trajs)
        refined_match = refined_metrics["horizon_match"].type_as(
            refined_trajs
        )
        base_score_match = (
            base_match * score_weight
        ).sum() / score_weight.sum().clamp_min(self.eps)
        refined_score_match = (
            refined_match * score_weight
        ).sum() / score_weight.sum().clamp_min(self.eps)
        top1 = logits.argmax(dim=-1)
        batch_idx = torch.arange(
            logits.shape[0], device=logits.device
        )
        metrics = {
            "loss_deployed_refiner": total,
            "loss_refiner_score_aligned": loss_score_aligned,
            "loss_refiner_coverage": loss_coverage,
            "loss_refiner_ade": loss_ade,
            "loss_refiner_no_harm": loss_no_harm,
            "loss_refiner_diversity": loss_diversity,
            "loss_refiner_delta_reg": loss_delta_reg,
            "loss_refiner_delta_smooth": loss_delta_smooth,
            "refiner_base_score_cost": base_score_cost,
            "refiner_final_score_cost": refined_score_cost,
            "refiner_score_cost_gain": (
                base_score_cost - refined_score_cost
            ),
            "refiner_base_score_match": base_score_match,
            "refiner_final_score_match": refined_score_match,
            "refiner_base_top1_cost": base_weighted_cost[
                batch_idx, top1
            ].mean(),
            "refiner_final_top1_cost": (
                (
                    refined_cost * valid_horizon_weight[:, None]
                ).sum(dim=-1)
                / valid_denominator[:, None]
            )[batch_idx, top1].mean(),
            "refiner_oracle_ade": refined_metrics["ade"].amin(
                dim=1
            ).mean(),
            "refiner_oracle_fde": refined_metrics["fde"].amin(
                dim=1
            ).mean(),
            "joint_oracle_ade": refined_metrics["ade"].amin(
                dim=1
            ).mean(),
            "joint_oracle_fde": refined_metrics["fde"].amin(
                dim=1
            ).mean(),
            "refiner_delta_abs": dense_delta.abs().mean(),
            "refiner_delta_max": dense_delta.abs().amax(),
            "refiner_gate": refiner_output["gate"].mean(),
            "refiner_worse_horizon_rate": (
                (refined_cost > base_cost + allowed_regression)
                .type_as(refined_trajs)
                * valid[:, None]
            ).sum()
            / (
                valid[:, None].expand_as(refined_cost).sum().clamp_min(1.0)
            ),
        }
        return total, metrics

    def _improvement_gated_residual_flow_loss(
        self, final_output, input_dict
    ):
        output = final_output.get("improvement_gated_output")
        if output is None:
            raise RuntimeError(
                "Missing improvement-gated output during JFER-v2 training"
            )
        module = self.candidate_residual_flow
        cfg = module.improvement_cfg
        deployed = output["trajectories"]
        base = output["base_trajectories"].detach()
        proposal = output["proposal_trajectories"]
        logits = final_output["improvement_gated_logits"].detach()
        deployed_metrics = self._official_match_quality(deployed, input_dict)
        with torch.no_grad():
            base_metrics = self._official_match_quality(base, input_dict)
            proposal_metrics = self._official_match_quality(
                proposal.detach(), input_dict
            )

        valid_bool = deployed_metrics["pair_valid"].bool()
        valid = valid_bool.type_as(deployed)
        horizon_weight = module.horizon_weights.type_as(deployed)[None]
        valid_horizon_weight = valid * horizon_weight
        valid_denominator = valid_horizon_weight.sum(
            dim=-1
        ).clamp_min(self.eps)
        sample_weight = self._pair_sample_weights(
            input_dict, deployed.device, deployed.dtype
        )
        probability = torch.softmax(
            logits / max(float(cfg.get("SCORE_TEMPERATURE", 1.0)), self.eps),
            dim=-1,
        )

        deployed_cost = deployed_metrics["horizon_cost"]
        base_cost = base_metrics["horizon_cost"].detach()
        proposal_cost = proposal_metrics["horizon_cost"].detach()
        target_cost = float(cfg.get("TARGET_NORMALIZED_COST", 0.90))
        threshold_temperature = max(
            float(cfg.get("THRESHOLD_TEMPERATURE", 0.12)), self.eps
        )
        score_penalty = threshold_temperature * F.softplus(
            (deployed_cost - target_cost) / threshold_temperature
        )
        score_per_sample = (
            score_penalty
            * probability[:, :, None]
            * valid_horizon_weight[:, None]
        ).sum(dim=(1, 2)) / valid_denominator
        loss_deployed_geometry = self._weighted_batch_mean(
            score_per_sample, sample_weight
        )

        coverage_temperature = max(
            float(cfg.get("COVERAGE_TEMPERATURE", 0.20)), self.eps
        )
        softmin_cost = -coverage_temperature * (
            torch.logsumexp(
                -deployed_cost / coverage_temperature, dim=1
            )
            - math.log(max(deployed_cost.shape[1], 1))
        )
        coverage_penalty = threshold_temperature * F.softplus(
            (softmin_cost - target_cost) / threshold_temperature
        )
        coverage_per_sample = (
            coverage_penalty * valid_horizon_weight
        ).sum(dim=-1) / valid_denominator
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        responsibility_temperature = max(
            float(cfg.get("RESPONSIBILITY_TEMPERATURE", 0.50)), self.eps
        )
        base_weighted_cost = (
            base_cost * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_denominator[:, None]
        responsibility = torch.softmax(
            -base_weighted_cost / responsibility_temperature, dim=-1
        )
        loss_ade = self._weighted_batch_mean(
            (responsibility * deployed_metrics["ade"]).sum(dim=-1),
            sample_weight,
        )

        no_harm_tolerance = float(cfg.get("NO_HARM_TOLERANCE", 0.02))
        no_harm = F.relu(
            deployed_cost - base_cost - no_harm_tolerance
        )
        no_harm_per_sample = (
            no_harm * valid_horizon_weight[:, None]
        ).sum(dim=(1, 2)) / (
            valid_denominator * deployed_cost.shape[1]
        )
        loss_no_harm = self._weighted_batch_mean(
            no_harm_per_sample, sample_weight
        )

        gain_target = (base_cost - proposal_cost).detach()
        gain_margin = float(cfg.get("GAIN_POSITIVE_MARGIN", 0.0))
        beneficial = gain_target > gain_margin
        valid_candidate_horizon = valid_bool[:, None].expand_as(gain_target)
        gain_denominator = valid_candidate_horizon.sum().clamp_min(1)
        positive_weight = deployed.new_tensor(
            float(cfg.get("GAIN_POSITIVE_WEIGHT", 2.0))
        )
        gain_classification = F.binary_cross_entropy_with_logits(
            output["alpha_logits"],
            beneficial.type_as(deployed),
            reduction="none",
            pos_weight=positive_weight,
        )
        loss_gain_classification = (
            gain_classification
            * valid_candidate_horizon.type_as(deployed)
        ).sum() / gain_denominator
        clipped_gain = gain_target.clamp(
            -float(cfg.get("GAIN_TARGET_CLIP", 2.0)),
            float(cfg.get("GAIN_TARGET_CLIP", 2.0)),
        )
        gain_regression = F.smooth_l1_loss(
            output["gain_prediction"], clipped_gain, reduction="none"
        )
        loss_gain_regression = (
            gain_regression
            * valid_candidate_horizon.type_as(deployed)
        ).sum() / gain_denominator

        alpha_horizon = output["alpha_horizon"]
        harmful = (~beneficial) & valid_candidate_horizon
        loss_false_open = (
            alpha_horizon * harmful.type_as(alpha_horizon)
        ).sum() / harmful.sum().clamp_min(1)
        deployed_delta = output["deployed_delta"]
        loss_residual_regularization = deployed_delta.square().mean()
        delta_velocity = deployed_delta[:, :, :, 1:] - deployed_delta[
            :, :, :, :-1
        ]
        delta_acceleration = (
            delta_velocity[:, :, :, 1:]
            - delta_velocity[:, :, :, :-1]
        )
        loss_smoothness = delta_acceleration.square().mean()
        loss_alpha_smoothness = (
            alpha_horizon[:, :, 1:] - alpha_horizon[:, :, :-1]
        ).square().mean()

        phase = int(output["phase"])
        geometry_loss = (
            float(cfg.get("LOSS_WEIGHT_DEPLOYED_GEOMETRY", 1.0))
            * loss_deployed_geometry
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 1.0)) * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_ADE", 0.10)) * loss_ade
            + float(cfg.get("LOSS_WEIGHT_NO_HARM", 2.0)) * loss_no_harm
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL", 0.01))
            * loss_residual_regularization
            + float(cfg.get("LOSS_WEIGHT_SMOOTHNESS", 0.02))
            * loss_smoothness
        )
        gate_loss = (
            float(cfg.get("LOSS_WEIGHT_GAIN_CLASSIFICATION", 1.0))
            * loss_gain_classification
            + float(cfg.get("LOSS_WEIGHT_GAIN_REGRESSION", 0.25))
            * loss_gain_regression
            + float(cfg.get("LOSS_WEIGHT_FALSE_OPEN", 1.0))
            * loss_false_open
            + float(cfg.get("LOSS_WEIGHT_ALPHA_SMOOTHNESS", 0.02))
            * loss_alpha_smoothness
        )
        if not output["effect_enabled"]:
            # Matched control executes the complete graph but cannot change
            # deployed trajectories, scores, membership, or confidence.
            total = (
                output["proposal_delta"].sum()
                + output["alpha_logits"].sum()
                + output["gain_prediction"].sum()
            ) * 0.0
        elif phase == 0:
            total = geometry_loss
        elif phase == 1:
            total = geometry_loss + gate_loss
        else:
            total = geometry_loss + gate_loss

        open_threshold = float(cfg.get("OPEN_THRESHOLD", 0.5))
        opened = (alpha_horizon >= open_threshold) & valid_candidate_horizon
        true_open = opened & beneficial
        harmful_open = opened & (~beneficial)
        open_count = opened.sum().clamp_min(1)
        harmful_count = harmful.sum().clamp_min(1)
        score_weight = (
            probability[:, :, None] * valid_horizon_weight[:, None]
        )
        metrics = {
            "loss_improvement_gated_flow": total,
            "loss_v2_deployed_geometry": loss_deployed_geometry,
            "loss_v2_coverage": loss_coverage,
            "loss_v2_ade": loss_ade,
            "loss_v2_no_harm": loss_no_harm,
            "loss_v2_gain_classification": loss_gain_classification,
            "loss_v2_gain_regression": loss_gain_regression,
            "loss_v2_false_open": loss_false_open,
            "loss_v2_residual_regularization": (
                loss_residual_regularization
            ),
            "loss_v2_smoothness": loss_smoothness,
            "loss_v2_alpha_smoothness": loss_alpha_smoothness,
            "v2_phase": deployed.new_tensor(float(phase)),
            "v2_alpha_mean": alpha_horizon.mean(),
            "v2_alpha_p50": alpha_horizon.detach().flatten().median(),
            "v2_open_rate": (
                opened.sum().type_as(deployed)
                / valid_candidate_horizon.sum().clamp_min(1)
            ),
            "v2_beneficial_precision": (
                true_open.sum().type_as(deployed) / open_count
            ),
            "v2_harmful_open_rate": (
                harmful_open.sum().type_as(deployed) / harmful_count
            ),
            "v2_harmful_fraction_among_open": (
                harmful_open.sum().type_as(deployed) / open_count
            ),
            "v2_proposal_beneficial_rate": (
                (beneficial & valid_candidate_horizon).sum().type_as(deployed)
                / valid_candidate_horizon.sum().clamp_min(1)
            ),
            "v2_gain_target_mean": (
                gain_target * valid_candidate_horizon.type_as(gain_target)
            ).sum() / gain_denominator,
            "v2_proposal_delta_abs": output["proposal_delta"].abs().mean(),
            "v2_deployed_delta_abs": deployed_delta.abs().mean(),
            "v2_deployed_delta_max": deployed_delta.abs().amax(),
            "v2_base_score_cost": (
                base_cost * score_weight
            ).sum() / score_weight.sum().clamp_min(self.eps),
            "v2_deployed_score_cost": (
                deployed_cost * score_weight
            ).sum() / score_weight.sum().clamp_min(self.eps),
            "v2_oracle_ade": deployed_metrics["ade"].amin(dim=1).mean(),
            "v2_oracle_fde": deployed_metrics["fde"].amin(dim=1).mean(),
            "joint_oracle_ade": deployed_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": deployed_metrics["fde"].amin(dim=1).mean(),
        }
        return total, metrics

    @torch.no_grad()
    def _set_to_slot_oracle_assignment(
        self, utility, trajectories, baseline_indices, slot_logits
    ):
        compatibility = self._selection_compatibility(trajectories)
        selected_rows = []
        for batch_idx in range(utility.shape[0]):
            selected = []
            for candidate_tensor in utility[batch_idx].argsort(
                descending=True
            ):
                candidate = int(candidate_tensor.item())
                if not selected or bool(
                    compatibility[
                        batch_idx, candidate, selected
                    ].all().item()
                ):
                    selected.append(candidate)
                    if len(selected) == self.num_output_modes:
                        break
            if len(selected) < self.num_output_modes:
                for candidate_tensor in utility[batch_idx].argsort(
                    descending=True
                ):
                    candidate = int(candidate_tensor.item())
                    if candidate not in selected:
                        selected.append(candidate)
                    if len(selected) == self.num_output_modes:
                        break
            selected_rows.append(selected)
        oracle_indices = torch.tensor(
            selected_rows,
            device=utility.device,
            dtype=torch.long,
        )
        slot_temperature = max(
            float(
                self.set_to_slot_transport_cfg.get(
                    "SLOT_VALUE_TEMPERATURE", 1.0
                )
            ),
            self.eps,
        )
        slot_weight = torch.softmax(
            slot_logits / slot_temperature, dim=-1
        )
        baseline_value = (
            utility.gather(1, baseline_indices) * slot_weight
        ).sum(dim=-1)
        oracle_value = (
            utility.gather(1, oracle_indices) * slot_weight
        ).sum(dim=-1)
        oracle_gain = oracle_value - baseline_value
        gain_margin = float(
            self.set_to_slot_transport_cfg.get(
                "ORACLE_GAIN_MARGIN", 0.02
            )
        )
        accept_target = oracle_gain > gain_margin
        target_indices = torch.where(
            accept_target[:, None], oracle_indices, baseline_indices
        )
        return target_indices, oracle_indices, oracle_gain, accept_target

    @torch.no_grad()
    def _sequential_mode_targets(
        self,
        utility,
        boundary,
        trajectories,
        baseline_indices,
        slot_weight,
    ):
        """Build a variable-cardinality replacement teacher from the baseline.

        The trusted deployed set is an explicit no-op action.  A candidate can
        enter the target set only when its marginal set utility beats the
        weakest current member, and the complete proposed set must improve the
        baseline objective.  This avoids imposing a fixed expansion quota on
        scenes whose protected candidates are already stronger.
        """
        cfg = self.sequential_mode_decoder_cfg
        module = self.sequential_mode_decoder
        horizon_weight = module.horizon_weights.type_as(utility)
        steps = module.measurement_steps.to(trajectories.device)
        signature = trajectories.index_select(3, steps).permute(
            0, 1, 3, 2, 4
        ).reshape(
            trajectories.shape[0], trajectories.shape[1], steps.numel(), -1
        )
        pair_distance = (
            signature[:, :, None] - signature[:, None, :]
        ).square().sum(dim=-1).mean(dim=-1).sqrt()
        diversity_scale = max(
            float(cfg.get("TARGET_DIVERSITY_SCALE", 3.0)), self.eps
        )
        coverage_weight = float(cfg.get("TARGET_COVERAGE_WEIGHT", 0.75))
        diversity_weight = float(cfg.get("TARGET_DIVERSITY_WEIGHT", 0.20))
        max_replacements = int(
            cfg.get("TARGET_MAX_REPLACEMENTS", self.num_output_modes)
        )
        max_replacements = max(
            0, min(max_replacements, self.num_output_modes)
        )
        marginal_gain_margin = float(
            cfg.get("TARGET_MARGINAL_GAIN_MARGIN", 0.02)
        )
        total_gain_margin = float(
            cfg.get("TARGET_TOTAL_GAIN_MARGIN", 0.02)
        )
        batch_size, num_candidates = utility.shape
        batch_index = torch.arange(
            batch_size, device=utility.device
        )[:, None]

        def set_value(indices, preserve_order):
            selected_utility = utility.gather(1, indices)
            if not preserve_order:
                selected_utility = selected_utility.sort(
                    dim=-1, descending=True
                ).values
            rank_value = (selected_utility * slot_weight).sum(dim=-1)
            selected_boundary = boundary[
                batch_index[:, :, None],
                indices[:, :, None],
                torch.arange(
                    boundary.shape[-1], device=boundary.device
                )[None, None],
            ]
            coverage_value = (
                selected_boundary.amax(dim=1) * horizon_weight[None]
            ).sum(dim=-1)
            selected_pair_distance = pair_distance.gather(
                1,
                indices[:, :, None].expand(
                    -1, -1, num_candidates
                ),
            ).gather(
                2,
                indices[:, None].expand(
                    -1, self.num_output_modes, -1
                ),
            )
            upper = torch.triu(
                torch.ones(
                    self.num_output_modes,
                    self.num_output_modes,
                    device=utility.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            diversity_value = selected_pair_distance.masked_select(
                upper[None].expand_as(selected_pair_distance)
            ).reshape(batch_size, -1).mean(dim=-1)
            diversity_value = diversity_value.div(
                diversity_scale
            ).clamp(max=2.0)
            return (
                rank_value
                + coverage_weight * coverage_value
                + diversity_weight * diversity_value
            )

        selected = baseline_indices.clone()
        active = torch.ones(
            batch_size, device=utility.device, dtype=torch.bool
        )
        for _ in range(max_replacements):
            selected_mask = torch.zeros(
                batch_size,
                num_candidates,
                device=utility.device,
                dtype=torch.bool,
            ).scatter(1, selected, True)
            selected_boundary = boundary[
                batch_index[:, :, None],
                selected[:, :, None],
                torch.arange(
                    boundary.shape[-1], device=boundary.device
                )[None, None],
            ]
            covered = selected_boundary.amax(dim=1)
            coverage_gain = (
                boundary
                * (1.0 - covered[:, None])
                * horizon_weight[None, None]
            ).sum(dim=-1)
            distance_to_selected = pair_distance.gather(
                2,
                selected[:, None].expand(-1, num_candidates, -1),
            ).amin(dim=-1)
            diversity_gain = distance_to_selected.div(
                diversity_scale
            ).clamp(max=2.0)
            proposal_score = (
                utility
                + coverage_weight * coverage_gain
                + diversity_weight * diversity_gain
            ).masked_fill(selected_mask, -torch.inf)
            proposal_value, proposal_index = proposal_score.max(dim=-1)
            selected_utility = utility.gather(1, selected)
            weakest_value, weakest_slot = selected_utility.min(dim=-1)
            accept = active & (
                proposal_value - weakest_value > marginal_gain_margin
            )
            replacement = selected.gather(
                1, weakest_slot[:, None]
            ).squeeze(-1)
            chosen = torch.where(accept, proposal_index, replacement)
            selected = selected.scatter(
                1, weakest_slot[:, None], chosen[:, None]
            )
            active = accept

        target_order = utility.gather(1, selected).argsort(
            dim=-1, descending=True
        )
        oracle_indices = selected.gather(1, target_order)
        baseline_value = set_value(
            baseline_indices, preserve_order=True
        )
        oracle_value = set_value(
            oracle_indices, preserve_order=True
        )
        oracle_gain = oracle_value - baseline_value
        accept_target = oracle_gain > total_gain_margin
        target_indices = torch.where(
            accept_target[:, None], oracle_indices, baseline_indices
        )
        baseline_membership = torch.zeros(
            batch_size,
            num_candidates,
            device=utility.device,
            dtype=torch.bool,
        ).scatter(1, baseline_indices, True)
        target_replacement_count = (
            ~baseline_membership.gather(1, target_indices)
        ).sum(dim=-1)
        return (
            target_indices,
            oracle_indices,
            oracle_gain,
            accept_target,
            target_replacement_count,
        )

    def _sequential_mode_loss(self, final_output, input_dict):
        sequential = final_output.get("sequential_mode_decoder")
        module = self.sequential_mode_decoder
        if sequential is None or module is None:
            raise RuntimeError("Missing sequential-mode decoder output")
        cfg = self.sequential_mode_decoder_cfg
        trajectories = final_output["joint_trajs"]
        metrics = self._official_match_quality(trajectories, input_dict)
        valid = metrics["pair_valid"].type_as(trajectories)
        cost = metrics["horizon_cost"]
        match = metrics["horizon_match"].type_as(trajectories)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=cost.detach(),
            pair_valid=metrics["pair_valid"],
        )
        boundary_temperature = max(
            float(cfg.get("BOUNDARY_TEMPERATURE", 0.20)), self.eps
        )
        boundary = torch.sigmoid(
            (1.0 - cost.detach()) / boundary_temperature
        )
        horizon_weight = module.horizon_weights.type_as(trajectories)
        horizon_utility = (
            float(cfg.get("UTILITY_CREDIT_WEIGHT", 2.0))
            * credit["credited_target"].type_as(trajectories)
            + float(cfg.get("UTILITY_MATCH_WEIGHT", 0.5)) * match.detach()
            + float(cfg.get("UTILITY_BOUNDARY_WEIGHT", 1.0)) * boundary
            - float(cfg.get("UTILITY_COST_WEIGHT", 0.25))
            * cost.detach().clamp_max(4.0)
        )
        utility = (
            horizon_utility * valid[:, None] * horizon_weight[None, None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)[:, None]

        branch_ids = (
            torch.arange(trajectories.shape[1], device=trajectories.device)[
                None
            ]
            .expand(trajectories.shape[0], -1)
            .ge(self.num_output_modes)
            .long()
        )
        slot_weight = torch.softmax(
            sequential["slot_logits"].detach(), dim=-1
        )
        (
            target_indices,
            oracle_indices,
            oracle_gain,
            accept_target,
            target_replacement_count,
        ) = self._sequential_mode_targets(
            utility=utility.detach(),
            boundary=boundary,
            trajectories=trajectories.detach(),
            baseline_indices=sequential["baseline_indices"],
            slot_weight=slot_weight,
        )
        raw_logits = sequential["raw_selection_logits"]
        teacher_used = torch.zeros_like(
            raw_logits[:, 0], dtype=torch.bool
        )
        teacher_logits = []
        for slot_idx in range(self.num_output_modes):
            teacher_logits.append(
                raw_logits[:, slot_idx].masked_fill(teacher_used, -1e4)
            )
            # Preserve each teacher-forcing mask version retained by
            # masked_fill for selector-loss backpropagation.
            teacher_used = teacher_used.scatter(
                1, target_indices[:, slot_idx, None], True
            )
        teacher_logits = torch.stack(teacher_logits, dim=1)
        assignment_ce = F.cross_entropy(
            (
                teacher_logits
                / max(
                    float(cfg.get("SELECTION_LOSS_TEMPERATURE", 0.7)),
                    self.eps,
                )
            ).reshape(-1, teacher_logits.shape[-1]),
            target_indices.reshape(-1),
            reduction="none",
        ).reshape_as(target_indices)
        sample_weight = self._pair_sample_weights(
            input_dict, trajectories.device, trajectories.dtype
        )
        loss_selection = self._weighted_batch_mean(
            (
                assignment_ce
                * (0.5 + self.num_output_modes * slot_weight)
            ).mean(dim=-1),
            sample_weight,
        )

        candidate_horizon_logits = sequential["candidate_horizon_logits"]
        target = credit["credited_target"].type_as(candidate_horizon_logits)
        supervision = credit["supervision_mask"].type_as(
            candidate_horizon_logits
        )
        horizon_bce = F.binary_cross_entropy_with_logits(
            candidate_horizon_logits, target, reduction="none"
        )
        positive_weight = float(cfg.get("CREDIT_POSITIVE_WEIGHT", 4.0))
        horizon_bce = horizon_bce * torch.where(
            target > 0.5,
            horizon_bce.new_full(horizon_bce.shape, positive_weight),
            torch.ones_like(horizon_bce),
        )
        horizon_loss_weight = (
            supervision * valid[:, None] * horizon_weight[None, None]
        )
        horizon_per_sample = (
            horizon_bce * horizon_loss_weight
        ).sum(dim=(1, 2)) / horizon_loss_weight.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_horizon = self._weighted_batch_mean(
            horizon_per_sample, sample_weight
        )

        target_temperature = max(
            float(cfg.get("VALUE_TARGET_TEMPERATURE", 0.5)), self.eps
        )
        value_temperature = max(
            float(cfg.get("VALUE_LOSS_TEMPERATURE", 0.7)), self.eps
        )
        target_probability = torch.softmax(
            utility.detach() / target_temperature, dim=-1
        )
        value_listwise = -(
            target_probability
            * F.log_softmax(
                sequential["candidate_value_logits"] / value_temperature,
                dim=-1,
            )
        ).sum(dim=-1)
        loss_value = self._weighted_batch_mean(
            value_listwise, sample_weight
        )

        assignment_probability = sequential["soft_assignment"]
        normalized_utility = utility.detach() - utility.detach().mean(
            dim=-1, keepdim=True
        )
        normalized_utility = normalized_utility / utility.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        expected_utility = torch.einsum(
            "bsc,bc->bs", assignment_probability, normalized_utility
        )
        loss_expected_utility = self._weighted_batch_mean(
            -(expected_utility * slot_weight).sum(dim=-1), sample_weight
        )

        inclusion_probability = 1.0 - (
            1.0 - assignment_probability.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        soft_success = 1.0 - (
            1.0
            - inclusion_probability[:, :, None]
            * boundary.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        coverage_per_sample = -(
            soft_success.clamp_min(self.eps).log()
            * valid
            * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        candidate_is_baseline = sequential[
            "candidate_is_baseline"
        ]
        target_membership = torch.zeros_like(
            candidate_is_baseline
        ).scatter(1, target_indices, True)
        admission_target = (
            target_membership & ~candidate_is_baseline
        ).type_as(trajectories)
        admission_mask = (~candidate_is_baseline).type_as(trajectories)
        admission_bce = F.binary_cross_entropy_with_logits(
            sequential["candidate_admission_logits"],
            admission_target,
            reduction="none",
        )
        admission_bce = admission_bce * torch.where(
            admission_target > 0.5,
            admission_bce.new_full(
                admission_bce.shape,
                float(cfg.get("ADMISSION_POSITIVE_WEIGHT", 4.0)),
            ),
            torch.ones_like(admission_bce),
        )
        admission_per_sample = (
            admission_bce * admission_mask
        ).sum(dim=-1) / admission_mask.sum(dim=-1).clamp_min(1.0)
        loss_admission = self._weighted_batch_mean(
            admission_per_sample, sample_weight
        )
        admission_probability = sequential[
            "candidate_admission_probability"
        ]
        admission_brier_per_sample = (
            (admission_probability - admission_target).square()
            * admission_mask
        ).sum(dim=-1) / admission_mask.sum(dim=-1).clamp_min(1.0)
        loss_admission_brier = self._weighted_batch_mean(
            admission_brier_per_sample, sample_weight
        )
        loss_admission = loss_admission + float(
            cfg.get("ADMISSION_BRIER_WEIGHT", 0.25)
        ) * loss_admission_brier

        normalized_assignment = F.normalize(
            assignment_probability, p=2, dim=-1
        )
        similarity = torch.matmul(
            normalized_assignment,
            normalized_assignment.transpose(1, 2),
        )
        off_diagonal = ~torch.eye(
            self.num_output_modes,
            device=similarity.device,
            dtype=torch.bool,
        )[None]
        loss_uniqueness = similarity.masked_select(
            off_diagonal.expand_as(similarity)
        ).mean()
        expected_replacement_count = (
            inclusion_probability
            * (~candidate_is_baseline).type_as(inclusion_probability)
        ).sum(dim=-1)
        loss_replacement_count = self._weighted_batch_mean(
            (
                expected_replacement_count
                - target_replacement_count.type_as(
                    expected_replacement_count
                )
            ).square()
            / max(self.num_output_modes ** 2, 1),
            sample_weight,
        )

        selector_total = (
            float(cfg.get("LOSS_WEIGHT_SELECTION", 1.0)) * loss_selection
            + float(cfg.get("LOSS_WEIGHT_HORIZON", 0.5)) * loss_horizon
            + float(cfg.get("LOSS_WEIGHT_VALUE", 0.5)) * loss_value
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_UTILITY", 0.5))
            * loss_expected_utility
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 0.75)) * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_UNIQUENESS", 0.05))
            * loss_uniqueness
            + float(cfg.get("LOSS_WEIGHT_ADMISSION", 0.75))
            * loss_admission
            + float(cfg.get("LOSS_WEIGHT_REPLACEMENT_COUNT", 0.25))
            * loss_replacement_count
            + float(cfg.get("LOSS_WEIGHT_ASSIGNMENT_REG", 0.001))
            * sequential["assignment_residual"].square().mean()
        )

        selected_base = sequential["selected_trajectories"]
        refined = sequential["refined_trajectories"]
        base_selected_metrics = self._official_match_quality(
            selected_base, input_dict
        )
        refined_metrics = self._official_match_quality(refined, input_dict)
        base_cost = base_selected_metrics["horizon_cost"].detach()
        refined_cost = refined_metrics["horizon_cost"]
        valid_horizon_weight = valid * horizon_weight[None]
        valid_horizon_denominator = valid_horizon_weight.sum(
            dim=-1
        ).clamp_min(self.eps)
        base_mode_cost = (
            base_cost * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_horizon_denominator[:, None]
        refined_mode_cost = (
            refined_cost * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_horizon_denominator[:, None]

        # Dual-anchor responsibility is deliberately sparse.  The geometric
        # oracle protects best-of-six coverage, while a refinable top-confidence
        # anchor improves the trajectory that contributes most to AP.  Other
        # modes are no-op targets, so multimodal hypotheses are not all pulled
        # toward the single observed future.
        batch_size, num_modes = base_mode_cost.shape
        batch_index = torch.arange(
            batch_size, device=base_mode_cost.device
        )
        oracle_index = base_selected_metrics["quality"].detach().argmin(
            dim=-1
        )
        top1_index = torch.zeros_like(oracle_index)
        oracle_ade = base_selected_metrics["ade"].detach()[
            batch_index, oracle_index
        ]
        top1_ade = base_selected_metrics["ade"].detach()[
            batch_index, top1_index
        ]
        best_mode_cost = base_mode_cost.detach()[batch_index, oracle_index]
        top1_mode_cost = base_mode_cost.detach()[batch_index, top1_index]
        max_refinable_ade = float(cfg.get("REFINE_MAX_BASE_ADE", 4.0))
        oracle_refinable = oracle_ade <= max_refinable_ade
        top1_refinable = (
            (top1_ade <= max_refinable_ade)
            & (
                top1_mode_cost - best_mode_cost
                <= float(cfg.get("REFINE_TOP1_COST_MARGIN", 0.75))
            )
        )
        oracle_responsibility = F.one_hot(
            oracle_index, num_classes=num_modes
        ).type_as(refined) * oracle_refinable[:, None].type_as(refined)
        top1_responsibility = F.one_hot(
            top1_index, num_classes=num_modes
        ).type_as(refined) * top1_refinable[:, None].type_as(refined)
        responsibility = (
            float(cfg.get("REFINE_ORACLE_WEIGHT", 1.0))
            * oracle_responsibility
            + float(cfg.get("REFINE_TOP1_WEIGHT", 0.5))
            * top1_responsibility
        )
        responsibility_denominator = responsibility.sum(
            dim=-1
        ).clamp_min(1.0)
        normalized_responsibility = (
            responsibility / responsibility_denominator[:, None]
        )
        target_mode = responsibility > 0.0

        gt = base_selected_metrics["gt"][:, None]
        gt_mask = base_selected_metrics["mask"][:, None].type_as(refined)
        trajectory_regression = F.smooth_l1_loss(
            refined,
            gt.expand_as(refined),
            reduction="none",
            beta=float(cfg.get("REFINE_REGRESSION_BETA", 1.0)),
        ).sum(dim=-1)
        trajectory_regression = (
            trajectory_regression * gt_mask
        ).sum(dim=(-1, -2)) / gt_mask.sum(
            dim=(-1, -2)
        ).clamp_min(1.0)
        refine_target_per_sample = (
            trajectory_regression * normalized_responsibility
        ).sum(dim=-1)
        loss_refine_target = self._weighted_batch_mean(
            refine_target_per_sample, sample_weight
        )
        loss_refine_score = loss_refine_target

        # Direct local-coordinate teacher.  Regressing only the final dense
        # trajectory makes the gradient pass through interpolation, bounds and
        # a low-probability gate.  Project the observed residual into each
        # candidate's own tangent/normal frame and supervise the pre-gate
        # correction fraction directly.  This decouples correction geometry
        # from the separate decision of whether refinement should be active.
        sample_steps = module.sample_steps.to(refined.device)
        previous_steps = (sample_steps - 5).clamp_min(0)
        base_knot = selected_base.index_select(3, sample_steps).permute(
            0, 1, 3, 2, 4
        )
        previous_knot = selected_base.index_select(
            3, previous_steps
        ).permute(0, 1, 3, 2, 4)
        tangent = base_knot - previous_knot
        tangent_norm = torch.linalg.vector_norm(
            tangent, dim=-1, keepdim=True
        )
        fallback_tangent = torch.zeros_like(tangent)
        fallback_tangent[..., 0] = 1.0
        tangent = torch.where(
            tangent_norm > 1e-3,
            tangent / tangent_norm.clamp_min(1e-3),
            fallback_tangent,
        )
        normal = torch.stack(
            [-tangent[..., 1], tangent[..., 0]], dim=-1
        )
        gt_knot = base_selected_metrics["gt"].index_select(
            2, sample_steps
        ).permute(0, 2, 1, 3)[:, None]
        gt_knot_valid = base_selected_metrics["mask"].index_select(
            2, sample_steps
        ).permute(0, 2, 1)[:, None]
        world_teacher_delta = gt_knot - base_knot
        if module.refiner_coordinate_frame == "frenet":
            local_teacher_delta = torch.stack(
                [
                    (world_teacher_delta * tangent).sum(dim=-1),
                    (world_teacher_delta * normal).sum(dim=-1),
                ],
                dim=-1,
            )
        else:
            local_teacher_delta = world_teacher_delta
        max_local_delta = module.temporal_max_delta.type_as(refined)[
            None, None, :, None, :
        ].clamp_min(self.eps)
        teacher_clip_ratio = float(
            cfg.get("REFINE_LOCAL_TEACHER_CLIP_RATIO", 0.95)
        )
        local_teacher_fraction = (
            local_teacher_delta / max_local_delta
        ).clamp(-teacher_clip_ratio, teacher_clip_ratio).detach()
        uses_residual_mixture = (
            sequential["refinement_branch_logits"] is not None
        )
        uses_counterfactual_gain = (
            uses_residual_mixture
            and module.refinement_selection_mode == "counterfactual_gain"
        )
        uses_hierarchical_gain = (
            uses_residual_mixture
            and module.refinement_selection_mode == "hierarchical_gain"
        )
        uses_branch_world_energy = (
            uses_residual_mixture
            and module.refinement_selection_mode == "branch_world_energy"
        )
        uses_absolute_gain_gate = (
            uses_counterfactual_gain
            or uses_hierarchical_gain
            or uses_branch_world_energy
        )
        loss_refine_branch = refined.new_zeros(())
        loss_refine_branch_listwise = refined.new_zeros(())
        loss_refine_gain_regression = refined.new_zeros(())
        loss_refine_gain_admission = refined.new_zeros(())
        loss_refine_expert_choice = refined.new_zeros(())
        branch_target = torch.zeros(
            batch_size,
            num_modes,
            device=refined.device,
            dtype=torch.long,
        )
        branch_accuracy = refined.new_zeros(())
        selected_non_noop_rate = refined.new_zeros(())
        selected_non_noop_target_rate = refined.new_zeros(())
        branch_oracle_gain = refined.new_zeros(())
        branch_selected_gain = refined.new_zeros(())
        branch_predicted_max_gain = refined.new_zeros(())
        branch_target_max_gain = refined.new_zeros(())
        branch_target_non_noop_rate = refined.new_zeros(())
        branch_false_positive_rate = refined.new_zeros(())
        branch_false_negative_rate = refined.new_zeros(())
        branch_rank_accuracy = refined.new_zeros(())
        branch_target_entropy = refined.new_zeros(())
        branch_energy_spread = refined.new_zeros(())

        if uses_residual_mixture:
            num_experts = module.num_refinement_experts
            knot_valid = gt_knot_valid.expand(
                -1, num_modes, -1, -1
            )
            knot_valid_float = knot_valid[..., None].type_as(refined)
            scene_local_residual = (
                local_teacher_fraction * knot_valid_float
            ).sum(dim=(2, 3)) / knot_valid_float.sum(
                dim=(2, 3)
            ).clamp_min(1.0)
            dominant_longitudinal = (
                scene_local_residual[:, :, 0].abs()
                >= scene_local_residual[:, :, 1].abs()
            )
            directional_expert = torch.where(
                dominant_longitudinal,
                torch.where(
                    scene_local_residual[:, :, 0] >= 0.0,
                    torch.zeros_like(branch_target),
                    torch.ones_like(branch_target),
                ),
                torch.where(
                    scene_local_residual[:, :, 1] >= 0.0,
                    torch.full_like(branch_target, 2),
                    torch.full_like(branch_target, 3),
                ),
            )
            branch_target_min_fraction = float(
                cfg.get("REFINE_BRANCH_TARGET_MIN_FRACTION", 0.05)
            )
            branch_needed = (
                scene_local_residual.abs().amax(dim=-1)
                > branch_target_min_fraction
            ) & target_mode
            expert_assignment_target = F.one_hot(
                directional_expert,
                num_classes=num_experts,
            ).type_as(refined) * branch_needed[:, :, None].type_as(refined)
            expert_teacher_fraction = local_teacher_fraction[:, :, None]
            expert_raw_fraction = sequential[
                "refinement_expert_raw_fraction"
            ]
            expert_fraction_error = F.smooth_l1_loss(
                expert_raw_fraction,
                expert_teacher_fraction.expand_as(expert_raw_fraction),
                reduction="none",
                beta=float(cfg.get("REFINE_LOCAL_TEACHER_BETA", 0.10)),
            )
            expert_teacher_weight = (
                normalized_responsibility[:, :, None, None, None, None]
                * expert_assignment_target[:, :, :, None, None, None]
                * gt_knot_valid[:, :, None, :, :, None].type_as(refined)
            )
            expert_teacher_per_sample = (
                expert_fraction_error * expert_teacher_weight
            ).sum(dim=(1, 2, 3, 4, 5)) / (
                expert_teacher_weight.sum(dim=(1, 2, 3, 4, 5)).clamp_min(1.0)
                * expert_fraction_error.shape[-1]
            )
            loss_refine_local_teacher = self._weighted_batch_mean(
                expert_teacher_per_sample, sample_weight
            )
            expert_output_fraction = (
                sequential["refinement_expert_local_delta"]
                / max_local_delta[:, :, None]
            )
            expert_output_error = F.smooth_l1_loss(
                expert_output_fraction,
                expert_teacher_fraction.expand_as(expert_output_fraction),
                reduction="none",
                beta=float(cfg.get("REFINE_LOCAL_OUTPUT_BETA", 0.10)),
            )
            expert_output_per_sample = (
                expert_output_error * expert_teacher_weight
            ).sum(dim=(1, 2, 3, 4, 5)) / (
                expert_teacher_weight.sum(dim=(1, 2, 3, 4, 5)).clamp_min(1.0)
                * expert_output_error.shape[-1]
            )
            loss_refine_local_output = self._weighted_batch_mean(
                expert_output_per_sample, sample_weight
            )

            branch_logits = sequential["refinement_branch_logits"]
            num_branches = num_experts + 1
            branch_trajectories = sequential[
                "refinement_branch_trajectories"
            ]
            branch_metrics = self._official_match_quality(
                branch_trajectories.detach().reshape(
                    batch_size,
                    num_modes * num_branches,
                    *branch_trajectories.shape[3:],
                ),
                input_dict,
            )
            branch_quality = branch_metrics["quality"].reshape(
                batch_size, num_modes, num_branches
            ).detach()
            branch_match = branch_metrics["joint_match"].reshape(
                batch_size, num_modes, num_branches
            ).type_as(branch_quality)
            branch_credit = (
                -branch_quality
                + float(cfg.get("REFINE_BRANCH_MATCH_BONUS", 1.0))
                * branch_match
            )
            expert_gain_target = (
                branch_credit[..., 1:] - branch_credit[..., :1]
            )
            gain_margin = float(
                cfg.get(
                    "REFINE_GAIN_TARGET_MARGIN",
                    module.refinement_gain_threshold,
                )
            )
            target_best_gain, target_best_expert = (
                expert_gain_target.max(dim=-1)
            )
            branch_target = torch.where(
                target_best_gain > gain_margin,
                target_best_expert + 1,
                torch.zeros_like(target_best_expert),
            )
            if uses_branch_world_energy:
                branch_energy_logits = sequential[
                    "refinement_branch_energy_logits"
                ]
                branch_temperature = max(
                    float(
                        cfg.get(
                            "REFINE_BRANCH_ENERGY_TEMPERATURE",
                            0.20,
                        )
                    ),
                    self.eps,
                )
                branch_soft_target = torch.softmax(
                    branch_credit / branch_temperature,
                    dim=-1,
                )
                branch_listwise = -(
                    branch_soft_target
                    * torch.log_softmax(branch_energy_logits, dim=-1)
                ).sum(dim=-1)
                branch_mode_weight = (
                    target_mode.type_as(refined)
                    + float(cfg.get("REFINE_GAIN_NON_TARGET_WEIGHT", 0.25))
                    * (~target_mode).type_as(refined)
                )
                branch_listwise_per_sample = (
                    branch_listwise * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_branch_listwise = self._weighted_batch_mean(
                    branch_listwise_per_sample,
                    sample_weight,
                )

                predicted_expert_gain = (
                    branch_energy_logits[..., 1:]
                    - branch_energy_logits[..., :1]
                )
                gain_target_clip = float(
                    cfg.get("REFINE_GAIN_TARGET_CLIP", 1.5)
                )
                clipped_gain_target = expert_gain_target.clamp(
                    -gain_target_clip,
                    gain_target_clip,
                )
                gain_regression = F.smooth_l1_loss(
                    predicted_expert_gain,
                    clipped_gain_target,
                    reduction="none",
                    beta=float(
                        cfg.get("REFINE_GAIN_REGRESSION_BETA", 0.05)
                    ),
                ).mean(dim=-1)
                gain_regression_per_sample = (
                    gain_regression * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_gain_regression = self._weighted_batch_mean(
                    gain_regression_per_sample,
                    sample_weight,
                )

                predicted_max_gain = sequential[
                    "refinement_predicted_max_gain"
                ]
                gain_positive = (
                    target_best_gain > gain_margin
                ).type_as(predicted_max_gain)
                gain_temperature = max(
                    float(cfg.get("REFINE_GAIN_LOGIT_TEMPERATURE", 0.10)),
                    self.eps,
                )
                gain_admission = F.binary_cross_entropy_with_logits(
                    (predicted_max_gain - gain_margin) / gain_temperature,
                    gain_positive,
                    reduction="none",
                )
                gain_admission_per_sample = (
                    gain_admission * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_gain_admission = self._weighted_batch_mean(
                    gain_admission_per_sample,
                    sample_weight,
                )
                with torch.no_grad():
                    branch_oracle_index = branch_credit.argmax(dim=-1)
                    branch_rank_prediction = branch_energy_logits.argmax(
                        dim=-1
                    )
                    supervised_count = branch_mode_weight.sum().clamp_min(1.0)
                    branch_rank_accuracy = (
                        (branch_rank_prediction == branch_oracle_index)
                        .type_as(refined)
                        * branch_mode_weight
                    ).sum() / supervised_count
                    branch_target_entropy = (
                        -(
                            branch_soft_target
                            * branch_soft_target.clamp_min(self.eps).log()
                        ).sum(dim=-1)
                        * branch_mode_weight
                    ).sum() / supervised_count
                    branch_energy_spread = (
                        branch_energy_logits.std(
                            dim=-1, unbiased=False
                        )
                        * branch_mode_weight
                    ).sum() / supervised_count
                loss_refine_branch = (
                    float(
                        cfg.get(
                            "LOSS_WEIGHT_REFINE_BRANCH_LISTWISE",
                            1.0,
                        )
                    )
                    * loss_refine_branch_listwise
                    + float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_REG", 0.5))
                    * loss_refine_gain_regression
                    + float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_CLS", 0.5))
                    * loss_refine_gain_admission
                )
            elif uses_hierarchical_gain:
                predicted_max_gain = sequential[
                    "refinement_predicted_max_gain"
                ]
                gain_target_clip = float(
                    cfg.get("REFINE_GAIN_TARGET_CLIP", 1.5)
                )
                max_gain_target = target_best_gain.clamp(
                    -gain_target_clip, gain_target_clip
                )
                gain_regression = F.smooth_l1_loss(
                    predicted_max_gain,
                    max_gain_target,
                    reduction="none",
                    beta=float(cfg.get("REFINE_GAIN_REGRESSION_BETA", 0.05)),
                )
                gain_positive = (
                    target_best_gain > gain_margin
                ).type_as(predicted_max_gain)
                gain_temperature = max(
                    float(cfg.get("REFINE_GAIN_LOGIT_TEMPERATURE", 0.10)),
                    self.eps,
                )
                gain_admission = F.binary_cross_entropy_with_logits(
                    (predicted_max_gain - gain_margin) / gain_temperature,
                    gain_positive,
                    reduction="none",
                )
                gain_admission_weight = (
                    gain_positive
                    * float(cfg.get("REFINE_GAIN_POSITIVE_WEIGHT", 1.0))
                    + (1.0 - gain_positive)
                    * float(cfg.get("REFINE_GAIN_NEGATIVE_WEIGHT", 1.0))
                )
                branch_mode_weight = (
                    target_mode.type_as(refined)
                    + float(cfg.get("REFINE_GAIN_NON_TARGET_WEIGHT", 0.25))
                    * (~target_mode).type_as(refined)
                )
                gain_weight = branch_mode_weight * gain_admission_weight
                gain_regression_per_sample = (
                    gain_regression * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                gain_admission_per_sample = (
                    gain_admission * gain_weight
                ).sum(dim=-1) / gain_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_gain_regression = self._weighted_batch_mean(
                    gain_regression_per_sample, sample_weight
                )
                loss_refine_gain_admission = self._weighted_batch_mean(
                    gain_admission_per_sample, sample_weight
                )

                choice_temperature = max(
                    float(
                        cfg.get("REFINE_EXPERT_CHOICE_TEMPERATURE", 0.20)
                    ),
                    self.eps,
                )
                expert_choice_target = torch.softmax(
                    expert_gain_target / choice_temperature, dim=-1
                )
                expert_choice_ce = -(
                    expert_choice_target
                    * torch.log_softmax(
                        sequential["refinement_expert_choice_logits"],
                        dim=-1,
                    )
                ).sum(dim=-1)
                choice_mode_weight = branch_mode_weight * gain_positive
                expert_choice_per_sample = (
                    expert_choice_ce * choice_mode_weight
                ).sum(dim=-1) / choice_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_expert_choice = self._weighted_batch_mean(
                    expert_choice_per_sample, sample_weight
                )
                loss_refine_branch = (
                    float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_REG", 1.0))
                    * loss_refine_gain_regression
                    + float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_CLS", 0.5))
                    * loss_refine_gain_admission
                    + float(cfg.get("LOSS_WEIGHT_REFINE_EXPERT_CHOICE", 0.5))
                    * loss_refine_expert_choice
                )
            elif uses_counterfactual_gain:
                # Learn an absolute counterfactual utility gain over no-op.
                # A relative five-way softmax must always choose a branch and
                # caused the measured 99% expert activation.  Absolute gains
                # give deployment a calibrated abstention decision instead.
                gain_target_clip = float(
                    cfg.get("REFINE_GAIN_TARGET_CLIP", 1.5)
                )
                clipped_gain_target = expert_gain_target.clamp(
                    -gain_target_clip, gain_target_clip
                )
                predicted_gain = branch_logits[..., 1:]
                gain_regression = F.smooth_l1_loss(
                    predicted_gain,
                    clipped_gain_target,
                    reduction="none",
                    beta=float(cfg.get("REFINE_GAIN_REGRESSION_BETA", 0.05)),
                ).mean(dim=-1)
                gain_positive = (
                    expert_gain_target > gain_margin
                ).type_as(predicted_gain)
                gain_temperature = max(
                    float(cfg.get("REFINE_GAIN_LOGIT_TEMPERATURE", 0.10)),
                    self.eps,
                )
                gain_admission = F.binary_cross_entropy_with_logits(
                    (predicted_gain - gain_margin) / gain_temperature,
                    gain_positive,
                    reduction="none",
                )
                gain_admission_weight = (
                    gain_positive
                    * float(cfg.get("REFINE_GAIN_POSITIVE_WEIGHT", 1.0))
                    + (1.0 - gain_positive)
                    * float(cfg.get("REFINE_GAIN_NEGATIVE_WEIGHT", 2.0))
                )
                gain_admission = (
                    gain_admission * gain_admission_weight
                ).sum(dim=-1) / gain_admission_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                branch_mode_weight = (
                    target_mode.type_as(refined)
                    + float(cfg.get("REFINE_GAIN_NON_TARGET_WEIGHT", 0.25))
                    * (~target_mode).type_as(refined)
                )
                gain_regression_per_sample = (
                    gain_regression * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                gain_admission_per_sample = (
                    gain_admission * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_gain_regression = self._weighted_batch_mean(
                    gain_regression_per_sample, sample_weight
                )
                loss_refine_gain_admission = self._weighted_batch_mean(
                    gain_admission_per_sample, sample_weight
                )
                loss_refine_branch = (
                    float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_REG", 1.0))
                    * loss_refine_gain_regression
                    + float(cfg.get("LOSS_WEIGHT_REFINE_GAIN_CLS", 0.5))
                    * loss_refine_gain_admission
                )
            else:
                branch_temperature = max(
                    float(
                        cfg.get("REFINE_BRANCH_UTILITY_TEMPERATURE", 0.25)
                    ),
                    self.eps,
                )
                branch_soft_target = torch.softmax(
                    branch_credit / branch_temperature, dim=-1
                )
                branch_target = branch_credit.argmax(dim=-1)
                branch_ce = -(
                    branch_soft_target
                    * torch.log_softmax(branch_logits, dim=-1)
                ).sum(dim=-1)
                if bool(
                    cfg.get("REFINE_BRANCH_SUPERVISE_TARGET_ONLY", False)
                ):
                    branch_mode_weight = target_mode.type_as(refined)
                else:
                    branch_mode_weight = (
                        target_mode.type_as(refined)
                        + float(cfg.get("REFINE_BRANCH_NOOP_WEIGHT", 1.0))
                        * (~target_mode).type_as(refined)
                    )
                branch_ce_per_sample = (
                    branch_ce * branch_mode_weight
                ).sum(dim=-1) / branch_mode_weight.sum(
                    dim=-1
                ).clamp_min(1.0)
                loss_refine_branch = self._weighted_batch_mean(
                    branch_ce_per_sample, sample_weight
                )
            with torch.no_grad():
                if uses_absolute_gain_gate:
                    branch_prediction = sequential[
                        "selected_refinement_branch"
                    ]
                else:
                    branch_prediction = branch_logits.argmax(dim=-1)
                branch_accuracy = (
                    (branch_prediction == branch_target).type_as(refined)
                    * branch_mode_weight
                ).sum() / branch_mode_weight.sum().clamp_min(1.0)
                selected_non_noop_rate = (
                    branch_prediction > 0
                ).type_as(refined).mean()
                selected_non_noop_target_rate = (
                    (branch_prediction > 0).type_as(refined)
                    * target_mode.type_as(refined)
                ).sum() / target_mode.sum().clamp_min(1).type_as(refined)
                target_mode_float = target_mode.type_as(refined)
                branch_oracle_gain = (
                    (
                        branch_quality[..., 0]
                        - branch_quality.amin(dim=-1)
                    )
                    * target_mode_float
                ).sum() / target_mode_float.sum().clamp_min(1.0)
                selected_branch_quality = branch_quality.gather(
                    -1, branch_prediction[..., None]
                ).squeeze(-1)
                branch_selected_gain = (
                    (
                        branch_quality[..., 0]
                        - selected_branch_quality
                    )
                    * target_mode_float
                ).sum() / target_mode_float.sum().clamp_min(1.0)
                if uses_absolute_gain_gate:
                    supervised_weight = branch_mode_weight
                    supervised_count = supervised_weight.sum().clamp_min(1.0)
                    branch_predicted_max_gain = (
                        sequential["refinement_predicted_max_gain"]
                        * supervised_weight
                    ).sum() / supervised_count
                    branch_target_max_gain = (
                        target_best_gain * supervised_weight
                    ).sum() / supervised_count
                    branch_target_non_noop_rate = (
                        (branch_target > 0).type_as(refined)
                        * supervised_weight
                    ).sum() / supervised_count
                    branch_false_positive_rate = (
                        (
                            (branch_prediction > 0)
                            & (branch_target == 0)
                        ).type_as(refined)
                        * supervised_weight
                    ).sum() / supervised_count
                    branch_false_negative_rate = (
                        (
                            (branch_prediction == 0)
                            & (branch_target > 0)
                        ).type_as(refined)
                        * supervised_weight
                    ).sum() / supervised_count

            supervised_teacher_fraction = (
                expert_teacher_fraction
                * expert_assignment_target[:, :, :, None, None, None]
            )
            local_teacher_weight = expert_teacher_weight
            raw_fraction_for_metric = expert_raw_fraction
        else:
            target_mode_float = target_mode[
                :, :, None, None, None
            ].type_as(refined)
            supervised_teacher_fraction = (
                local_teacher_fraction * target_mode_float
            )
            raw_fraction_error = F.smooth_l1_loss(
                sequential["raw_knot_fraction"],
                supervised_teacher_fraction,
                reduction="none",
                beta=float(cfg.get("REFINE_LOCAL_TEACHER_BETA", 0.10)),
            )
            negative_mode_weight = (~target_mode).type_as(refined)
            negative_mode_weight = negative_mode_weight / (
                negative_mode_weight.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0)
            )
            local_mode_weight = (
                normalized_responsibility
                + float(
                    cfg.get("REFINE_LOCAL_NON_TARGET_WEIGHT", 0.25)
                )
                * negative_mode_weight
            )
            local_teacher_weight = (
                local_mode_weight[:, :, None, None, None]
                * gt_knot_valid[:, :, :, :, None].type_as(refined)
            )
            local_teacher_per_sample = (
                raw_fraction_error * local_teacher_weight
            ).sum(dim=(1, 2, 3, 4)) / (
                local_teacher_weight.sum(
                    dim=(1, 2, 3, 4)
                ).clamp_min(1.0)
                * raw_fraction_error.shape[-1]
            )
            loss_refine_local_teacher = self._weighted_batch_mean(
                local_teacher_per_sample, sample_weight
            )

            gated_fraction = (
                sequential["local_knot_delta"] / max_local_delta
            )
            gated_fraction_error = F.smooth_l1_loss(
                gated_fraction,
                supervised_teacher_fraction,
                reduction="none",
                beta=float(cfg.get("REFINE_LOCAL_OUTPUT_BETA", 0.10)),
            )
            gated_teacher_per_sample = (
                gated_fraction_error * local_teacher_weight
            ).sum(dim=(1, 2, 3, 4)) / (
                local_teacher_weight.sum(
                    dim=(1, 2, 3, 4)
                ).clamp_min(1.0)
                * gated_fraction_error.shape[-1]
            )
            loss_refine_local_output = self._weighted_batch_mean(
                gated_teacher_per_sample, sample_weight
            )
            raw_fraction_for_metric = sequential["raw_knot_fraction"]

        improvement_margin = float(
            cfg.get("REFINE_IMPROVEMENT_MARGIN", 0.02)
        )
        improvement_penalty = F.relu(
            refined_mode_cost - base_mode_cost + improvement_margin
        )
        loss_refine_improvement = self._weighted_batch_mean(
            (
                improvement_penalty * normalized_responsibility
            ).sum(dim=-1),
            sample_weight,
        )

        softmin_temperature = max(
            float(cfg.get("REFINE_SOFTMIN_TEMPERATURE", 0.20)), self.eps
        )
        refined_softmin = -softmin_temperature * torch.logsumexp(
            -refined_cost / softmin_temperature, dim=1
        )
        base_softmin = -softmin_temperature * torch.logsumexp(
            -base_cost / softmin_temperature, dim=1
        )
        loss_refine_coverage = self._weighted_batch_mean(
            (
                refined_softmin * valid * horizon_weight[None]
            ).sum(dim=-1)
            / (valid * horizon_weight[None]).sum(dim=-1).clamp_min(self.eps),
            sample_weight,
        )
        no_harm = F.relu(
            refined_cost
            - base_cost
            - float(cfg.get("REFINE_NO_HARM_TOLERANCE", 0.01))
        )
        protected_mode_weight = (
            (~target_mode).type_as(refined)
            + float(cfg.get("REFINE_TARGET_NO_HARM_WEIGHT", 0.25))
            * target_mode.type_as(refined)
        )
        protected_mode_weight = protected_mode_weight * (
            0.5 + num_modes * slot_weight
        )
        no_harm_weight = (
            protected_mode_weight[:, :, None]
            * valid_horizon_weight[:, None]
        )
        loss_refine_no_harm = self._weighted_batch_mean(
            (no_harm * no_harm_weight).sum(dim=(1, 2))
            / no_harm_weight.sum(dim=(1, 2)).clamp_min(self.eps),
            sample_weight,
        )

        dense_delta = sequential["dense_delta"]
        if uses_residual_mixture:
            regularized_dense_delta = sequential[
                "refinement_expert_dense_delta"
            ]
        else:
            regularized_dense_delta = dense_delta
        loss_refine_reg = regularized_dense_delta.square().mean()
        delta_energy = dense_delta.square().mean(dim=(-1, -2, -3))
        preserve_weight = (~target_mode).type_as(delta_energy)
        loss_refine_preserve = self._weighted_batch_mean(
            (delta_energy * preserve_weight).sum(dim=-1)
            / preserve_weight.sum(dim=-1).clamp_min(1.0),
            sample_weight,
        )
        velocity_delta = (
            regularized_dense_delta[..., 1:, :]
            - regularized_dense_delta[..., :-1, :]
        )
        acceleration_delta = (
            velocity_delta[..., 1:, :] - velocity_delta[..., :-1, :]
        )
        loss_refine_smooth = acceleration_delta.square().mean()

        steps = module.measurement_steps.to(refined.device)
        base_signature = selected_base.index_select(3, steps).reshape(
            batch_size, num_modes, -1
        )
        refined_signature = refined.index_select(3, steps).reshape(
            batch_size, num_modes, -1
        )
        base_distance = torch.cdist(
            base_signature.float(), base_signature.float()
        ).type_as(refined).detach()
        refined_distance = torch.cdist(
            refined_signature.float(), refined_signature.float()
        ).type_as(refined)
        diversity_mask = torch.triu(
            torch.ones(
                num_modes,
                num_modes,
                device=refined.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None] & (
            base_distance
            > float(cfg.get("REFINE_DIVERSITY_MIN_BASE_DISTANCE", 0.5))
        )
        diversity_penalty = F.relu(
            float(cfg.get("REFINE_DIVERSITY_PRESERVE_RATIO", 0.90))
            * base_distance
            - refined_distance
        )
        diversity_per_sample = (
            diversity_penalty * diversity_mask.type_as(refined)
        ).sum(dim=(1, 2)) / diversity_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_refine_diversity = self._weighted_batch_mean(
            diversity_per_sample, sample_weight
        )

        if uses_residual_mixture:
            gate_target = (branch_target > 0).type_as(refined)
            loss_refine_gate = refined.new_zeros(())
        else:
            gate_delta_threshold = float(
                cfg.get("REFINE_GATE_TARGET_MIN_FRACTION", 0.05)
            )
            gate_needed = (
                local_teacher_fraction.abs().amax(dim=-1)
                > gate_delta_threshold
            ) & gt_knot_valid.expand(-1, num_modes, -1, -1)
            gate_target = (
                gate_needed.any(dim=-1)
                & target_mode[:, :, None]
            ).type_as(sequential["refine_gate_logits"])
            gate_bce = F.binary_cross_entropy_with_logits(
                sequential["refine_gate_logits"],
                gate_target,
                reduction="none",
            ).mean(dim=(1, 2))
            loss_refine_gate = self._weighted_batch_mean(
                gate_bce, sample_weight
            )

        refiner_total = (
            float(cfg.get("LOSS_WEIGHT_REFINE_LOCAL_TEACHER", 1.0))
            * loss_refine_local_teacher
            + float(cfg.get("LOSS_WEIGHT_REFINE_LOCAL_OUTPUT", 0.5))
            * loss_refine_local_output
            + float(cfg.get("LOSS_WEIGHT_REFINE_TARGET", 1.0))
            * loss_refine_target
            + float(cfg.get("LOSS_WEIGHT_REFINE_COVERAGE", 1.0))
            * loss_refine_coverage
            + float(cfg.get("LOSS_WEIGHT_REFINE_IMPROVEMENT", 0.5))
            * loss_refine_improvement
            + float(cfg.get("LOSS_WEIGHT_REFINE_NO_HARM", 2.0))
            * loss_refine_no_harm
            + float(cfg.get("LOSS_WEIGHT_REFINE_PRESERVE", 0.25))
            * loss_refine_preserve
            + float(cfg.get("LOSS_WEIGHT_REFINE_DIVERSITY", 0.10))
            * loss_refine_diversity
            + float(cfg.get("LOSS_WEIGHT_REFINE_GATE", 0.25))
            * loss_refine_gate
            + float(cfg.get("LOSS_WEIGHT_REFINE_BRANCH", 1.0))
            * loss_refine_branch
            + float(cfg.get("LOSS_WEIGHT_REFINE_REG", 0.01))
            * loss_refine_reg
            + float(cfg.get("LOSS_WEIGHT_REFINE_SMOOTH", 0.02))
            * loss_refine_smooth
        )

        if self.train_stage == "candidate_bank_sequential_mode_warmup":
            total = selector_total
        elif self.train_stage == "candidate_bank_sequential_mode_refine":
            total = refiner_total
        else:
            total = selector_total + float(
                cfg.get("LOSS_WEIGHT_REFINER", 1.0)
            ) * refiner_total

        selected = sequential["selected_indices"]
        target_recall = (
            selected[:, :, None] == target_indices[:, None]
        ).any(dim=1).float().mean()
        changed = (
            selected != sequential["baseline_indices"]
        ).any(dim=-1).float().mean()
        if uses_residual_mixture:
            if uses_absolute_gain_gate:
                predicted_max_gain = sequential[
                    "refinement_predicted_max_gain"
                ]
                gain_temperature = max(
                    float(cfg.get("REFINE_GAIN_LOGIT_TEMPERATURE", 0.10)),
                    self.eps,
                )
                per_mode_refine_probability = torch.sigmoid(
                    (
                        predicted_max_gain
                        - module.refinement_gain_threshold
                    )
                    / gain_temperature
                )
                refine_gate_probability = (
                    per_mode_refine_probability.mean()
                )
                target_refine_probability = (
                    per_mode_refine_probability
                    * target_mode.type_as(refined)
                ).sum() / target_mode.sum().clamp_min(1).type_as(refined)
            else:
                branch_probability = torch.softmax(
                    sequential["refinement_branch_logits"], dim=-1
                )
                refine_gate_probability = branch_probability[
                    ..., 1:
                ].sum(dim=-1).mean()
                target_refine_probability = (
                    branch_probability[..., 1:].sum(dim=-1)
                    * target_mode.type_as(refined)
                ).sum() / target_mode.sum().clamp_min(1).type_as(refined)
        else:
            per_mode_refine_probability = torch.sigmoid(
                sequential["refine_gate_logits"]
            ).mean(dim=-1)
            refine_gate_probability = per_mode_refine_probability.mean()
            target_refine_probability = (
                per_mode_refine_probability
                * target_mode.type_as(refined)
            ).sum() / target_mode.sum().clamp_min(1).type_as(refined)
        return total, {
            "loss_sequential_mode": total,
            "loss_sequential_selector": selector_total,
            "loss_sequential_selection": loss_selection,
            "loss_sequential_horizon": loss_horizon,
            "loss_sequential_value": loss_value,
            "loss_sequential_expected_utility": loss_expected_utility,
            "loss_sequential_coverage": loss_coverage,
            "loss_sequential_uniqueness": loss_uniqueness,
            "loss_sequential_admission": loss_admission,
            "loss_sequential_admission_brier": loss_admission_brier,
            "loss_sequential_replacement_count": loss_replacement_count,
            "loss_sequential_refiner": refiner_total,
            "loss_sequential_refine_score": loss_refine_score,
            "loss_sequential_refine_target": loss_refine_target,
            "loss_sequential_refine_local_teacher": (
                loss_refine_local_teacher
            ),
            "loss_sequential_refine_local_output": (
                loss_refine_local_output
            ),
            "loss_sequential_refine_coverage": loss_refine_coverage,
            "loss_sequential_refine_improvement": (
                loss_refine_improvement
            ),
            "loss_sequential_refine_no_harm": loss_refine_no_harm,
            "loss_sequential_refine_preserve": loss_refine_preserve,
            "loss_sequential_refine_diversity": loss_refine_diversity,
            "loss_sequential_refine_gate": loss_refine_gate,
            "loss_sequential_refine_branch": loss_refine_branch,
            "loss_sequential_refine_branch_listwise": (
                loss_refine_branch_listwise
            ),
            "loss_sequential_refine_gain_regression": (
                loss_refine_gain_regression
            ),
            "loss_sequential_refine_gain_admission": (
                loss_refine_gain_admission
            ),
            "loss_sequential_refine_expert_choice": (
                loss_refine_expert_choice
            ),
            "sequential_target_recall": target_recall,
            "sequential_changed_scene_rate": changed,
            "sequential_expansion_mode_rate": (
                selected >= self.num_output_modes
            ).float().mean(),
            "sequential_expected_expansion_count": (
                expected_replacement_count.mean()
            ),
            "sequential_target_replacement_count": (
                target_replacement_count.float().mean()
            ),
            "sequential_target_accept_rate": (
                accept_target.float().mean()
            ),
            "sequential_target_set_gain": oracle_gain.mean(),
            "sequential_admission_target_rate": (
                admission_target.sum()
                / admission_mask.sum().clamp_min(1.0)
            ),
            "sequential_admission_probability": (
                admission_probability * admission_mask
            ).sum() / admission_mask.sum().clamp_min(1.0),
            "sequential_admission_positive_probability": (
                admission_probability * admission_target
            ).sum() / admission_target.sum().clamp_min(1.0),
            "sequential_admission_negative_probability": (
                admission_probability
                * admission_mask
                * (1.0 - admission_target)
            ).sum() / (
                admission_mask * (1.0 - admission_target)
            ).sum().clamp_min(1.0),
            "sequential_admission_brier": loss_admission_brier,
            "sequential_admitted_candidate_rate": (
                sequential["hard_admitted"] & ~candidate_is_baseline
            ).float().sum() / admission_mask.sum().clamp_min(1.0),
            "sequential_assignment_delta_abs": sequential[
                "assignment_residual"
            ].abs().mean(),
            "sequential_value_gain": sequential["value_gain"].abs(),
            "sequential_refine_delta_abs": dense_delta.abs().mean(),
            "sequential_refine_delta_max": dense_delta.abs().amax(),
            "sequential_refine_target_mode_rate": (
                target_mode.float().mean()
            ),
            "sequential_refine_oracle_is_top1_rate": (
                oracle_index == top1_index
            ).float().mean(),
            "sequential_refine_top1_eligible_rate": (
                top1_refinable.float().mean()
            ),
            "sequential_refine_gate_probability": (
                refine_gate_probability
            ),
            "sequential_refine_target_probability": (
                target_refine_probability
            ),
            "sequential_refine_gate_target_rate": gate_target.mean(),
            "sequential_refine_branch_accuracy": branch_accuracy,
            "sequential_refine_selected_non_noop_rate": (
                selected_non_noop_rate
            ),
            "sequential_refine_selected_non_noop_target_rate": (
                selected_non_noop_target_rate
            ),
            "sequential_refine_branch_oracle_gain": branch_oracle_gain,
            "sequential_refine_branch_selected_gain": (
                branch_selected_gain
            ),
            "sequential_refine_branch_predicted_max_gain": (
                branch_predicted_max_gain
            ),
            "sequential_refine_branch_target_max_gain": (
                branch_target_max_gain
            ),
            "sequential_refine_branch_target_non_noop_rate": (
                branch_target_non_noop_rate
            ),
            "sequential_refine_branch_false_positive_rate": (
                branch_false_positive_rate
            ),
            "sequential_refine_branch_false_negative_rate": (
                branch_false_negative_rate
            ),
            "sequential_refine_branch_rank_accuracy": branch_rank_accuracy,
            "sequential_refine_branch_target_entropy": branch_target_entropy,
            "sequential_refine_branch_energy_spread": branch_energy_spread,
            "sequential_refine_target_long_positive_rate": (
                branch_target == 1
            ).float().mean(),
            "sequential_refine_target_long_negative_rate": (
                branch_target == 2
            ).float().mean(),
            "sequential_refine_target_lat_positive_rate": (
                branch_target == 3
            ).float().mean(),
            "sequential_refine_target_lat_negative_rate": (
                branch_target == 4
            ).float().mean(),
            "sequential_refine_teacher_fraction_abs": (
                supervised_teacher_fraction.abs() * local_teacher_weight
            ).sum() / (
                local_teacher_weight.sum().clamp_min(1.0)
                * supervised_teacher_fraction.shape[-1]
            ),
            "sequential_refine_raw_fraction_abs": (
                raw_fraction_for_metric.abs() * local_teacher_weight
            ).sum() / (
                local_teacher_weight.sum().clamp_min(1.0)
                * raw_fraction_for_metric.shape[-1]
            ),
            "sequential_refine_base_target_cost": (
                base_mode_cost * normalized_responsibility
            ).sum(dim=-1).mean(),
            "sequential_refine_final_target_cost": (
                refined_mode_cost * normalized_responsibility
            ).sum(dim=-1).mean(),
            "sequential_refine_base_top1_ade": (
                base_selected_metrics["ade"][:, 0].mean()
            ),
            "sequential_refine_final_top1_ade": (
                refined_metrics["ade"][:, 0].mean()
            ),
            "sequential_base_oracle_ade": base_selected_metrics["ade"].amin(
                dim=1
            ).mean(),
            "sequential_final_oracle_ade": refined_metrics["ade"].amin(
                dim=1
            ).mean(),
            "sequential_base_oracle_fde": base_selected_metrics["fde"].amin(
                dim=1
            ).mean(),
            "sequential_final_oracle_fde": refined_metrics["fde"].amin(
                dim=1
            ).mean(),
            "joint_oracle_ade": refined_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": refined_metrics["fde"].amin(dim=1).mean(),
        }

    def _set_to_slot_value_loss(
        self,
        transport,
        trajectories,
        input_dict,
        utility,
        horizon_utility,
        boundary,
        valid,
        horizon_weight,
    ):
        """Train candidate values and the gain of the deployed top-k action."""
        cfg = self.set_to_slot_transport_cfg
        value_logits = transport["candidate_value_logits"]
        horizon_logits = transport["candidate_horizon_logits"]
        sample_weight = self._pair_sample_weights(
            input_dict,
            trajectories.device,
            trajectories.dtype,
        )
        target_temperature = max(
            float(cfg.get("VALUE_TARGET_TEMPERATURE", 0.5)), self.eps
        )
        prediction_temperature = max(
            float(cfg.get("VALUE_LOSS_TEMPERATURE", 0.7)), self.eps
        )

        horizon_target_probability = torch.softmax(
            horizon_utility.detach() / target_temperature, dim=1
        )
        horizon_log_probability = torch.log_softmax(
            horizon_logits / prediction_temperature, dim=1
        )
        horizon_listwise = -(
            horizon_target_probability * horizon_log_probability
        ).sum(dim=1)
        horizon_listwise_per_sample = (
            horizon_listwise * valid * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_horizon_listwise = self._weighted_batch_mean(
            horizon_listwise_per_sample, sample_weight
        )

        target_probability = torch.softmax(
            utility.detach() / target_temperature, dim=-1
        )
        value_log_probability = torch.log_softmax(
            value_logits / prediction_temperature, dim=-1
        )
        listwise_per_sample = -(
            target_probability * value_log_probability
        ).sum(dim=-1)
        loss_listwise = self._weighted_batch_mean(
            listwise_per_sample, sample_weight
        )

        target_difference = (
            utility.detach()[:, :, None] - utility.detach()[:, None, :]
        )
        predicted_difference = (
            value_logits[:, :, None] - value_logits[:, None, :]
        )
        pair_margin = float(cfg.get("VALUE_PAIRWISE_MARGIN", 0.10))
        pair_mask = target_difference.abs() > pair_margin
        pair_mask = pair_mask & torch.triu(
            torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
        )
        pair_direction = target_difference.sign()
        pairwise_term = F.softplus(
            -pair_direction * predicted_difference
        )
        pairwise_per_sample = (
            pairwise_term * pair_mask.type_as(pairwise_term)
        ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)
        loss_pairwise = self._weighted_batch_mean(
            pairwise_per_sample, sample_weight
        )

        assignment_probability = transport["soft_assignment"]
        expected_utility = torch.einsum(
            "bsc,bc->bs", assignment_probability, utility.detach()
        )
        slot_weight = torch.softmax(
            transport["slot_logits"].detach(), dim=-1
        )
        expected_utility_per_sample = -(
            expected_utility * slot_weight
        ).sum(dim=-1)
        loss_expected_utility = self._weighted_batch_mean(
            expected_utility_per_sample, sample_weight
        )

        inclusion_probability = 1.0 - (
            1.0 - assignment_probability.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        soft_horizon_success = 1.0 - (
            1.0
            - inclusion_probability[:, :, None]
            * boundary.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        coverage_per_sample = -(
            soft_horizon_success.clamp_min(self.eps).log()
            * valid
            * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        normalized_assignment = F.normalize(
            assignment_probability, p=2, dim=-1
        )
        assignment_similarity = torch.matmul(
            normalized_assignment,
            normalized_assignment.transpose(1, 2),
        )
        off_diagonal = ~torch.eye(
            self.num_output_modes,
            device=assignment_similarity.device,
            dtype=torch.bool,
        )[None]
        loss_uniqueness = assignment_similarity.masked_select(
            off_diagonal.expand_as(assignment_similarity)
        ).mean()

        proposed_indices = transport["proposed_indices"]
        baseline_indices = transport["baseline_indices"]
        proposal_value = (
            utility.detach().gather(1, proposed_indices) * slot_weight
        ).sum(dim=-1)
        baseline_value = (
            utility.detach().gather(1, baseline_indices) * slot_weight
        ).sum(dim=-1)
        proposal_gain = proposal_value - baseline_value
        proposal_gain_margin = float(
            cfg.get("PROPOSAL_GAIN_MARGIN", 0.01)
        )
        acceptance_target = proposal_gain > proposal_gain_margin
        acceptance_bce = F.binary_cross_entropy_with_logits(
            transport["acceptance_logit"],
            acceptance_target.type_as(transport["acceptance_logit"]),
            reduction="none",
        )
        positive_weight = float(cfg.get("ACCEPTANCE_POS_WEIGHT", 1.0))
        negative_weight = float(cfg.get("ACCEPTANCE_NEG_WEIGHT", 2.0))
        acceptance_weight = torch.where(
            acceptance_target,
            acceptance_bce.new_full(acceptance_bce.shape, positive_weight),
            acceptance_bce.new_full(acceptance_bce.shape, negative_weight),
        )
        loss_acceptance = self._weighted_batch_mean(
            acceptance_bce * acceptance_weight, sample_weight
        )
        gain_scale = max(
            float(cfg.get("PROPOSAL_GAIN_SCALE", 1.0)), self.eps
        )
        predicted_gain = gain_scale * torch.tanh(
            transport["acceptance_logit"]
        )
        loss_gain = self._weighted_batch_mean(
            F.smooth_l1_loss(
                predicted_gain,
                proposal_gain.clamp(-gain_scale, gain_scale),
                reduction="none",
                beta=0.1,
            ),
            sample_weight,
        )

        proposal_hard = F.one_hot(
            proposed_indices, num_classes=utility.shape[1]
        ).type_as(assignment_probability)
        proposal_assignment = (
            proposal_hard
            + assignment_probability
            - assignment_probability.detach()
        )
        proposal_trajectories = torch.einsum(
            "bsc,bcath->bsath", proposal_assignment, trajectories.detach()
        )
        proposal_metrics = self._official_match_quality(
            proposal_trajectories, input_dict
        )
        proposal_cost = proposal_metrics["horizon_cost"]
        geometry_per_sample = (
            proposal_cost
            * slot_weight[:, :, None]
            * valid[:, None]
            * horizon_weight[None, None]
        ).sum(dim=(1, 2)) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_geometry = self._weighted_batch_mean(
            geometry_per_sample, sample_weight
        )

        baseline_trajectories = OnlineSetToSlotTransport._gather_candidates(
            trajectories.detach(), baseline_indices
        )
        baseline_metrics = self._official_match_quality(
            baseline_trajectories, input_dict
        )
        softmin_temperature = max(
            float(cfg.get("SOFTMIN_TEMPERATURE", 0.20)), self.eps
        )
        proposal_softmin = -softmin_temperature * (
            torch.logsumexp(
                -proposal_cost / softmin_temperature, dim=1
            )
            - math.log(self.num_output_modes)
        )
        baseline_softmin = -softmin_temperature * (
            torch.logsumexp(
                -baseline_metrics["horizon_cost"].detach()
                / softmin_temperature,
                dim=1,
            )
            - math.log(self.num_output_modes)
        )
        no_harm = F.relu(
            proposal_softmin
            - baseline_softmin
            - float(cfg.get("NO_HARM_TOLERANCE", 0.02))
        )
        no_harm_per_sample = (
            no_harm * valid * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_no_harm = self._weighted_batch_mean(
            no_harm_per_sample, sample_weight
        )

        value_residual = transport.get("value_residual")
        loss_residual_reg = (
            value_residual.square().mean()
            if value_residual is not None
            else value_logits.new_zeros(())
        )
        total = (
            float(cfg.get("LOSS_WEIGHT_HORIZON_LISTWISE", 1.0))
            * loss_horizon_listwise
            + float(cfg.get("LOSS_WEIGHT_VALUE_LISTWISE", 1.0))
            * loss_listwise
            + float(cfg.get("LOSS_WEIGHT_VALUE_PAIRWISE", 0.5))
            * loss_pairwise
            + float(cfg.get("LOSS_WEIGHT_ACCEPTANCE", 0.5))
            * loss_acceptance
            + float(cfg.get("LOSS_WEIGHT_GAIN", 0.25)) * loss_gain
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_UTILITY", 0.5))
            * loss_expected_utility
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 0.5))
            * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_GEOMETRY", 0.25))
            * loss_geometry
            + float(cfg.get("LOSS_WEIGHT_NO_HARM", 0.5))
            * loss_no_harm
            + float(cfg.get("LOSS_WEIGHT_UNIQUENESS", 0.05))
            * loss_uniqueness
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL_REG", 0.001))
            * loss_residual_reg
        )

        _, oracle_indices, oracle_gain, _ = (
            self._set_to_slot_oracle_assignment(
                utility=utility.detach(),
                trajectories=trajectories.detach(),
                baseline_indices=baseline_indices,
                slot_logits=transport["slot_logits"].detach(),
            )
        )
        oracle_recall = (
            proposed_indices[:, :, None] == oracle_indices[:, None]
        ).any(dim=1).float().mean()
        changed = (proposed_indices != baseline_indices).any(dim=-1)
        pairwise_correct = (
            (predicted_difference * target_difference > 0)
            & pair_mask
        ).sum().type_as(value_logits) / pair_mask.sum().clamp_min(1)
        accepted = transport["accepted"]
        acceptance_accuracy = (
            accepted == acceptance_target
        ).float().mean()
        return total, {
            "loss_set_to_slot_transport": total,
            "loss_set_to_slot_assignment": loss_listwise,
            "loss_set_to_slot_horizon_listwise": loss_horizon_listwise,
            "loss_set_to_slot_value_listwise": loss_listwise,
            "loss_set_to_slot_value_pairwise": loss_pairwise,
            "loss_set_to_slot_acceptance": loss_acceptance,
            "loss_set_to_slot_gain": loss_gain,
            "loss_set_to_slot_expected_utility": loss_expected_utility,
            "loss_set_to_slot_coverage": loss_coverage,
            "loss_set_to_slot_geometry": loss_geometry,
            "loss_set_to_slot_no_harm": loss_no_harm,
            "loss_set_to_slot_uniqueness": loss_uniqueness,
            "loss_set_to_slot_residual_reg": loss_residual_reg,
            "set_to_slot_oracle_gain": oracle_gain.mean(),
            "set_to_slot_proposal_gain": proposal_gain.mean(),
            "set_to_slot_positive_proposal_gain": (
                proposal_gain.clamp_min(0).mean()
            ),
            "set_to_slot_accept_target_rate": acceptance_target.float().mean(),
            "set_to_slot_accept_probability": transport[
                "acceptance_probability"
            ].mean(),
            "set_to_slot_accepted_rate": accepted.float().mean(),
            "set_to_slot_acceptance_accuracy": acceptance_accuracy,
            "set_to_slot_changed_scene_rate": (
                transport["hard_indices"] != baseline_indices
            ).any(dim=-1).float().mean(),
            "set_to_slot_proposal_changed_scene_rate": changed.float().mean(),
            "set_to_slot_oracle_recall": oracle_recall,
            "set_to_slot_value_pairwise_accuracy": pairwise_correct,
            "set_to_slot_expansion_rate": (
                transport["hard_indices"] >= self.num_output_modes
            ).float().mean(),
            "set_to_slot_proposal_expansion_rate": (
                proposed_indices >= self.num_output_modes
            ).float().mean(),
            "set_to_slot_residual_abs": (
                value_residual.abs().mean()
                if value_residual is not None
                else value_logits.new_zeros(())
            ),
            "set_to_slot_selected_oracle_ade": proposal_metrics[
                "ade"
            ].amin(dim=1).mean(),
            "set_to_slot_selected_oracle_fde": proposal_metrics[
                "fde"
            ].amin(dim=1).mean(),
            "joint_oracle_ade": proposal_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": proposal_metrics["fde"].amin(dim=1).mean(),
        }

    def _set_to_slot_transport_loss(self, final_output, input_dict):
        transport = final_output.get("set_to_slot_transport")
        if transport is None:
            raise RuntimeError(
                "Missing set-to-slot transport output during joint training"
            )
        trajectories = final_output["joint_trajs"]
        metrics = self._official_match_quality(trajectories, input_dict)
        valid = metrics["pair_valid"].type_as(trajectories)
        horizon_weight = self.set_to_slot_transport.horizon_weights.type_as(
            trajectories
        )
        cost = metrics["horizon_cost"]
        match = metrics["horizon_match"].type_as(trajectories)
        credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=cost.detach(),
            pair_valid=metrics["pair_valid"],
        )
        boundary_temperature = max(
            float(
                self.set_to_slot_transport_cfg.get(
                    "BOUNDARY_TEMPERATURE", 0.20
                )
            ),
            self.eps,
        )
        boundary = torch.sigmoid(
            (1.0 - cost.detach()) / boundary_temperature
        )
        horizon_utility = (
            float(
                self.set_to_slot_transport_cfg.get(
                    "UTILITY_CREDIT_WEIGHT", 2.0
                )
            )
            * credit["credited_target"].type_as(trajectories)
            + float(
                self.set_to_slot_transport_cfg.get(
                    "UTILITY_MATCH_WEIGHT", 0.5
                )
            )
            * match.detach()
            + float(
                self.set_to_slot_transport_cfg.get(
                    "UTILITY_BOUNDARY_WEIGHT", 1.0
                )
            )
            * boundary
            - float(
                self.set_to_slot_transport_cfg.get(
                    "UTILITY_COST_WEIGHT", 0.25
                )
            )
            * cost.detach().clamp_max(4.0)
        )
        utility = (
            horizon_utility
            * valid[:, None]
            * horizon_weight[None, None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)[:, None]

        if transport.get("candidate_value_logits") is not None:
            return self._set_to_slot_value_loss(
                transport=transport,
                trajectories=trajectories,
                input_dict=input_dict,
                utility=utility,
                horizon_utility=horizon_utility,
                boundary=boundary,
                valid=valid,
                horizon_weight=horizon_weight,
            )

        (
            target_indices,
            oracle_indices,
            oracle_gain,
            accept_target,
        ) = self._set_to_slot_oracle_assignment(
            utility=utility.detach(),
            trajectories=trajectories.detach(),
            baseline_indices=transport["baseline_indices"],
            slot_logits=transport["slot_logits"].detach(),
        )

        proposal_logits = transport["proposal_assignment_logits"]
        assignment_ce = F.cross_entropy(
            proposal_logits.reshape(-1, proposal_logits.shape[-1]),
            target_indices.reshape(-1),
            reduction="none",
        ).reshape_as(target_indices)
        slot_weight = torch.softmax(
            transport["slot_logits"].detach(), dim=-1
        )
        assignment_per_sample = (
            assignment_ce * (0.5 + self.num_output_modes * slot_weight)
        ).mean(dim=-1)
        sample_weight = self._pair_sample_weights(
            input_dict,
            trajectories.device,
            trajectories.dtype,
        )
        loss_assignment = self._weighted_batch_mean(
            assignment_per_sample, sample_weight
        )

        acceptance_bce = F.binary_cross_entropy_with_logits(
            transport["acceptance_logit"],
            accept_target.type_as(transport["acceptance_logit"]),
            reduction="none",
        )
        positive_weight = float(
            self.set_to_slot_transport_cfg.get(
                "ACCEPTANCE_POS_WEIGHT", 2.0
            )
        )
        acceptance_weight = torch.where(
            accept_target,
            acceptance_bce.new_full(acceptance_bce.shape, positive_weight),
            torch.ones_like(acceptance_bce),
        )
        loss_acceptance = self._weighted_batch_mean(
            acceptance_bce * acceptance_weight, sample_weight
        )

        assignment_probability = torch.softmax(
            proposal_logits
            / max(
                float(
                    self.set_to_slot_transport_cfg.get(
                        "LOSS_ASSIGNMENT_TEMPERATURE", 0.7
                    )
                ),
                self.eps,
            ),
            dim=-1,
        )
        normalized_utility = utility.detach() - utility.detach().mean(
            dim=-1, keepdim=True
        )
        normalized_utility = normalized_utility / utility.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        expected_utility = torch.einsum(
            "bsc,bc->bs", assignment_probability, normalized_utility
        )
        utility_per_sample = -(
            expected_utility * slot_weight
        ).sum(dim=-1)
        loss_expected_utility = self._weighted_batch_mean(
            utility_per_sample, sample_weight
        )

        inclusion_probability = 1.0 - (
            1.0 - assignment_probability.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        soft_horizon_success = 1.0 - (
            1.0
            - inclusion_probability[:, :, None]
            * boundary.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        coverage_per_sample = -(
            soft_horizon_success.clamp_min(self.eps).log()
            * valid
            * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        normalized_assignment = F.normalize(
            assignment_probability, p=2, dim=-1
        )
        similarity = torch.matmul(
            normalized_assignment,
            normalized_assignment.transpose(1, 2),
        )
        off_diagonal = ~torch.eye(
            self.num_output_modes,
            device=similarity.device,
            dtype=torch.bool,
        )[None]
        loss_uniqueness = similarity.masked_select(
            off_diagonal.expand_as(similarity)
        ).mean()

        selected_metrics = self._official_match_quality(
            transport["selected_trajectories"], input_dict
        )
        selected_cost = selected_metrics["horizon_cost"]
        score_probability = torch.softmax(
            transport["slot_logits"].detach(), dim=-1
        )
        score_cost_per_sample = (
            selected_cost
            * score_probability[:, :, None]
            * valid[:, None]
            * horizon_weight[None, None]
        ).sum(dim=(1, 2)) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_geometry = self._weighted_batch_mean(
            score_cost_per_sample, sample_weight
        )

        baseline_trajs = OnlineSetToSlotTransport._gather_candidates(
            trajectories.detach(), transport["baseline_indices"]
        )
        baseline_metrics = self._official_match_quality(
            baseline_trajs, input_dict
        )
        softmin_temperature = max(
            float(
                self.set_to_slot_transport_cfg.get(
                    "SOFTMIN_TEMPERATURE", 0.20
                )
            ),
            self.eps,
        )
        selected_softmin = -softmin_temperature * (
            torch.logsumexp(
                -selected_cost / softmin_temperature, dim=1
            )
            - math.log(self.num_output_modes)
        )
        baseline_softmin = -softmin_temperature * (
            torch.logsumexp(
                -baseline_metrics["horizon_cost"].detach()
                / softmin_temperature,
                dim=1,
            )
            - math.log(self.num_output_modes)
        )
        no_harm = F.relu(
            selected_softmin
            - baseline_softmin
            - float(
                self.set_to_slot_transport_cfg.get(
                    "NO_HARM_TOLERANCE", 0.02
                )
            )
        )
        no_harm_per_sample = (
            no_harm * valid * horizon_weight[None]
        ).sum(dim=-1) / (
            valid * horizon_weight[None]
        ).sum(dim=-1).clamp_min(self.eps)
        loss_no_harm = self._weighted_batch_mean(
            no_harm_per_sample, sample_weight
        )
        loss_residual_reg = transport[
            "assignment_residual"
        ].square().mean()

        cfg = self.set_to_slot_transport_cfg
        total = (
            float(cfg.get("LOSS_WEIGHT_ASSIGNMENT", 1.0))
            * loss_assignment
            + float(cfg.get("LOSS_WEIGHT_ACCEPTANCE", 0.5))
            * loss_acceptance
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_UTILITY", 0.5))
            * loss_expected_utility
            + float(cfg.get("LOSS_WEIGHT_COVERAGE", 0.5))
            * loss_coverage
            + float(cfg.get("LOSS_WEIGHT_GEOMETRY", 0.25))
            * loss_geometry
            + float(cfg.get("LOSS_WEIGHT_NO_HARM", 0.5))
            * loss_no_harm
            + float(cfg.get("LOSS_WEIGHT_UNIQUENESS", 0.05))
            * loss_uniqueness
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL_REG", 0.001))
            * loss_residual_reg
        )

        target_recall = (
            transport["proposed_indices"][:, :, None]
            == target_indices[:, None]
        ).any(dim=1).float().mean()
        oracle_recall = (
            transport["proposed_indices"][:, :, None]
            == oracle_indices[:, None]
        ).any(dim=1).float().mean()
        changed = (
            transport["hard_indices"]
            != transport["baseline_indices"]
        ).any(dim=-1)
        proposal_changed = (
            transport["proposed_indices"]
            != transport["baseline_indices"]
        ).any(dim=-1)
        return total, {
            "loss_set_to_slot_transport": total,
            "loss_set_to_slot_assignment": loss_assignment,
            "loss_set_to_slot_acceptance": loss_acceptance,
            "loss_set_to_slot_expected_utility": loss_expected_utility,
            "loss_set_to_slot_coverage": loss_coverage,
            "loss_set_to_slot_geometry": loss_geometry,
            "loss_set_to_slot_no_harm": loss_no_harm,
            "loss_set_to_slot_uniqueness": loss_uniqueness,
            "loss_set_to_slot_residual_reg": loss_residual_reg,
            "set_to_slot_oracle_gain": oracle_gain.mean(),
            "set_to_slot_accept_target_rate": accept_target.float().mean(),
            "set_to_slot_accept_probability": transport[
                "acceptance_probability"
            ].mean(),
            "set_to_slot_accepted_rate": transport[
                "accepted"
            ].float().mean(),
            "set_to_slot_changed_scene_rate": changed.float().mean(),
            "set_to_slot_proposal_changed_scene_rate": (
                proposal_changed.float().mean()
            ),
            "set_to_slot_target_recall": target_recall,
            "set_to_slot_oracle_recall": oracle_recall,
            "set_to_slot_expansion_rate": (
                transport["hard_indices"] >= self.num_output_modes
            ).float().mean(),
            "set_to_slot_proposal_expansion_rate": (
                transport["proposed_indices"] >= self.num_output_modes
            ).float().mean(),
            "set_to_slot_residual_abs": transport[
                "assignment_residual"
            ].abs().mean(),
            "set_to_slot_selected_oracle_ade": selected_metrics[
                "ade"
            ].amin(dim=1).mean(),
            "set_to_slot_selected_oracle_fde": selected_metrics[
                "fde"
            ].amin(dim=1).mean(),
            "joint_oracle_ade": selected_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": selected_metrics["fde"].amin(dim=1).mean(),
        }

    def _interaction_time_warp_loss(self, final_output, input_dict):
        time_warp = final_output.get("interaction_time_warp_bank")
        module = self.interaction_time_warp_bank
        if time_warp is None or module is None:
            raise RuntimeError(
                "Final output is missing interaction time-warp candidates"
            )
        cfg = self.interaction_time_warp_cfg
        num_base = self.num_output_modes
        num_policies = module.num_policies
        num_donors = module.num_base_modes
        base_trajs = final_output["joint_trajs"][:, :num_base].detach()
        expansion_trajs = time_warp["trajectories"]
        all_trajs = torch.cat([base_trajs, expansion_trajs], dim=1)
        base_metrics = self._official_match_quality(base_trajs, input_dict)
        expansion_metrics = self._official_match_quality(
            expansion_trajs, input_dict
        )
        all_metrics = self._official_match_quality(all_trajs, input_dict)

        all_donor_trajs = time_warp["all_donor_trajectories"]
        batch_size = all_donor_trajs.shape[0]
        flattened_donor_trajs = all_donor_trajs.reshape(
            batch_size,
            num_policies * num_donors,
            *all_donor_trajs.shape[3:],
        )
        donor_metrics_flat = self._official_match_quality(
            flattened_donor_trajs, input_dict
        )
        donor_metrics = {}
        for name, value in donor_metrics_flat.items():
            if torch.is_tensor(value) and value.ndim >= 2 and (
                value.shape[1] == num_policies * num_donors
            ):
                donor_metrics[name] = value.reshape(
                    batch_size,
                    num_policies,
                    num_donors,
                    *value.shape[2:],
                )
            else:
                donor_metrics[name] = value

        dtype = expansion_trajs.dtype
        device = expansion_trajs.device
        sample_weights = self._pair_sample_weights(
            input_dict, device, dtype
        )
        horizon_weights = module.horizon_weights.type_as(expansion_trajs)
        pair_valid = all_metrics["pair_valid"].type_as(expansion_trajs)
        valid_horizon_weight = pair_valid * horizon_weights[None]
        normalized_horizon_weight = valid_horizon_weight / (
            valid_horizon_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        )
        informative_scene = all_metrics["pair_valid"].any(
            dim=-1
        ).type_as(expansion_trajs)

        boundary_temperature = max(
            float(cfg.get("BOUNDARY_TEMPERATURE", 0.20)), self.eps
        )
        base_boundary = torch.sigmoid(
            (1.0 - base_metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        donor_boundary = torch.sigmoid(
            (1.0 - donor_metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        donor_match = donor_metrics["horizon_match"].type_as(
            expansion_trajs
        )
        base_covered = base_metrics["horizon_match"].any(dim=1)
        donor_rescue = donor_metrics["horizon_match"] & (
            ~base_covered[:, None, None]
        )
        base_best_cost = base_metrics["horizon_cost"].amin(
            dim=1
        ).detach()
        donor_gain = (
            base_best_cost[:, None, None]
            - donor_metrics["horizon_cost"].detach()
        ).clamp(-2.0, 2.0)
        donor_match_utility = (
            donor_match * normalized_horizon_weight[:, None, None]
        ).sum(dim=-1)
        donor_rescue_utility = (
            donor_rescue.type_as(expansion_trajs)
            * normalized_horizon_weight[:, None, None]
        ).sum(dim=-1)
        donor_boundary_utility = (
            donor_boundary * normalized_horizon_weight[:, None, None]
        ).sum(dim=-1)
        donor_gain_utility = (
            donor_gain.clamp_min(0.0)
            * normalized_horizon_weight[:, None, None]
        ).sum(dim=-1)
        donor_target_utility = (
            donor_match_utility
            + float(cfg.get("DONOR_RESCUE_WEIGHT", 1.5))
            * donor_rescue_utility
            + float(cfg.get("DONOR_BOUNDARY_WEIGHT", 0.25))
            * donor_boundary_utility
            + float(cfg.get("DONOR_GAIN_WEIGHT", 0.25))
            * donor_gain_utility
        ).detach()
        donor_target_temperature = max(
            float(cfg.get("DONOR_TARGET_TEMPERATURE", 0.15)), self.eps
        )
        donor_prediction_temperature = max(
            float(cfg.get("DONOR_PREDICTION_TEMPERATURE", 0.50)),
            self.eps,
        )
        donor_target_distribution = torch.softmax(
            donor_target_utility / donor_target_temperature, dim=-1
        )
        donor_log_probability = F.log_softmax(
            time_warp["donor_logits"] / donor_prediction_temperature,
            dim=-1,
        )
        donor_ce = -(
            donor_target_distribution * donor_log_probability
        ).sum(dim=-1).mean(dim=-1)
        loss_donor = self._weighted_batch_mean(
            donor_ce, sample_weights * informative_scene
        )

        # The donor target is detached, but the official ADE/FDE/miss quality
        # remains differentiable with respect to the bounded warp amplitude.
        donor_geometry = (
            donor_target_distribution
            * donor_metrics["quality"]
        ).sum(dim=-1).mean(dim=-1)
        geometry_scale = base_metrics["quality"].amin(
            dim=1
        ).detach().clamp_min(1.0)
        loss_geometry = self._weighted_batch_mean(
            donor_geometry / geometry_scale,
            sample_weights * informative_scene,
        )

        horizon_logits = time_warp["horizon_logits"]
        horizon_target = expansion_metrics["horizon_match"].type_as(
            horizon_logits
        )
        horizon_supervision = pair_valid[:, None]
        horizon_ce = F.binary_cross_entropy_with_logits(
            horizon_logits, horizon_target, reduction="none"
        )
        horizon_probability = torch.sigmoid(horizon_logits)
        horizon_p_t = (
            horizon_probability * horizon_target
            + (1.0 - horizon_probability) * (1.0 - horizon_target)
        )
        focal_alpha = float(cfg.get("FOCAL_ALPHA", 0.75))
        focal_gamma = float(cfg.get("FOCAL_GAMMA", 2.0))
        focal_alpha_t = (
            focal_alpha * horizon_target
            + (1.0 - focal_alpha) * (1.0 - horizon_target)
        )
        horizon_focal = (
            focal_alpha_t
            * (1.0 - horizon_p_t).pow(focal_gamma)
            * horizon_ce
        )
        horizon_weight = (
            horizon_supervision * horizon_weights[None, None]
        ).expand(-1, num_policies, -1)
        horizon_sample = (
            horizon_focal * horizon_weight
        ).sum(dim=(1, 2)) / horizon_weight.sum(dim=(1, 2)).clamp_min(1.0)
        loss_horizon = self._weighted_batch_mean(
            horizon_sample, sample_weights * informative_scene
        )

        gain_target = (
            base_best_cost[:, None]
            - expansion_metrics["horizon_cost"].detach()
        ).clamp(-2.0, 2.0)
        gain_loss = F.smooth_l1_loss(
            time_warp["gain"], gain_target, reduction="none", beta=0.25
        )
        gain_sample = (
            gain_loss * horizon_weight
        ).sum(dim=(1, 2)) / horizon_weight.sum(dim=(1, 2)).clamp_min(1.0)
        loss_gain = self._weighted_batch_mean(
            gain_sample, sample_weights * informative_scene
        )

        expansion_boundary = torch.sigmoid(
            (1.0 - expansion_metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        expansion_match_utility = (
            expansion_metrics["horizon_match"].type_as(expansion_trajs)
            * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        expansion_boundary_utility = (
            expansion_boundary * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        expansion_rescue = (
            expansion_metrics["horizon_match"]
            & ~base_covered[:, None]
        )
        expansion_rescue_utility = (
            expansion_rescue.type_as(expansion_trajs)
            * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        expansion_utility = (
            expansion_match_utility
            + float(cfg.get("ADMISSION_BOUNDARY_WEIGHT", 0.25))
            * expansion_boundary_utility
        )
        base_match_utility = (
            base_metrics["horizon_match"].type_as(expansion_trajs)
            * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        base_boundary_utility = (
            base_boundary * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        base_utility = (
            base_match_utility
            + float(cfg.get("ADMISSION_BOUNDARY_WEIGHT", 0.25))
            * base_boundary_utility
        )
        rank_order = time_warp["rank_order"]
        ranked_base_utility = torch.gather(base_utility, 1, rank_order)
        locked_count = max(
            num_base - min(self.max_expansion_replacements, num_policies),
            0,
        )
        replaceable_utility = ranked_base_utility[:, locked_count:]
        if replaceable_utility.shape[1] == 0:
            tail_floor = ranked_base_utility.amin(dim=-1, keepdim=True)
        else:
            tail_floor = replaceable_utility.amin(
                dim=-1, keepdim=True
            )

        compatibility = self._selection_compatibility(
            all_trajs.detach()
        )[:, num_base:, :num_base]
        if locked_count > 0:
            locked_indices = rank_order[:, :locked_count]
            locked_compatible = torch.gather(
                compatibility,
                2,
                locked_indices[:, None].expand(
                    -1, num_policies, -1
                ),
            ).all(dim=-1)
        else:
            locked_compatible = torch.ones(
                batch_size,
                num_policies,
                device=device,
                dtype=torch.bool,
            )
        admission_margin = float(cfg.get("ADMISSION_UTILITY_MARGIN", 0.02))
        admission_target = (
            (
                expansion_utility
                > tail_floor + admission_margin
            )
            | (expansion_rescue_utility > 0.0)
        ) & locked_compatible
        admission_target = admission_target.type_as(expansion_trajs)
        admission_logits = time_warp["admission_logits"]
        admission_ce = F.binary_cross_entropy_with_logits(
            admission_logits, admission_target, reduction="none"
        )
        admission_pos_weight = float(
            cfg.get("ADMISSION_POS_WEIGHT", 3.0)
        )
        admission_weight = 1.0 + (
            admission_pos_weight - 1.0
        ) * admission_target
        admission_sample = (
            admission_ce * admission_weight
        ).mean(dim=-1)
        loss_admission = self._weighted_batch_mean(
            admission_sample, sample_weights * informative_scene
        )

        credit = build_soft_map_credit_targets(
            horizon_match=all_metrics["horizon_match"],
            horizon_cost=all_metrics["horizon_cost"].detach(),
            pair_valid=all_metrics["pair_valid"],
        )
        credited_utility = (
            credit["credited_target"].type_as(expansion_trajs)
            * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        all_boundary = torch.sigmoid(
            (1.0 - all_metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        all_boundary_utility = (
            all_boundary * normalized_horizon_weight[:, None]
        ).sum(dim=-1)
        target_utility = (
            credited_utility
            + float(cfg.get("RANK_BOUNDARY_WEIGHT", 0.25))
            * all_boundary_utility
        ).detach()
        rank_target_temperature = max(
            float(cfg.get("RANK_TARGET_TEMPERATURE", 0.10)), self.eps
        )
        rank_prediction_temperature = max(
            float(cfg.get("RANK_PREDICTION_TEMPERATURE", 0.50)),
            self.eps,
        )
        target_distribution = torch.softmax(
            target_utility / rank_target_temperature, dim=-1
        )
        selection_logits = final_output[
            "protected_expansion_selection_logits"
        ]
        confidence_logits = final_output["joint_logits"]
        selection_listwise = -(
            target_distribution
            * F.log_softmax(
                selection_logits / rank_prediction_temperature, dim=-1
            )
        ).sum(dim=-1)
        confidence_listwise = -(
            target_distribution
            * F.log_softmax(
                confidence_logits / rank_prediction_temperature, dim=-1
            )
        ).sum(dim=-1)
        loss_selection_listwise = self._weighted_batch_mean(
            selection_listwise, sample_weights * informative_scene
        )
        loss_confidence_listwise = self._weighted_batch_mean(
            confidence_listwise, sample_weights * informative_scene
        )

        target_delta = (
            target_utility[:, num_base:, None]
            - target_utility[:, None, :num_base]
        )
        predicted_delta = (
            selection_logits[:, num_base:, None]
            - selection_logits[:, None, :num_base]
        ) / rank_prediction_temperature
        pair_margin = float(cfg.get("RANK_PAIR_TARGET_MARGIN", 0.02))
        pair_mask = target_delta.abs() > pair_margin
        pair_target = (target_delta > 0.0).type_as(expansion_trajs)
        pair_regret = target_delta.abs().clamp_min(pair_margin)
        pair_ce = F.binary_cross_entropy_with_logits(
            predicted_delta, pair_target, reduction="none"
        )
        pair_weight = pair_mask.type_as(expansion_trajs) * pair_regret
        pairwise_sample = (
            pair_ce * pair_weight
        ).sum(dim=(1, 2)) / pair_weight.sum(dim=(1, 2)).clamp_min(self.eps)
        loss_pairwise = self._weighted_batch_mean(
            pairwise_sample,
            sample_weights
            * pair_mask.any(dim=2).any(dim=1).type_as(expansion_trajs),
        )

        probability = torch.softmax(confidence_logits, dim=-1)
        group_ids = official_ap_group_ids(
            input_dict, batch_size, device
        )
        loss_global_ap = self._global_ap_surrogate(
            probability, credit, group_ids, scorer_cfg=cfg
        )

        base_shift = module.base_policy_shifts[
            None, :, None
        ].type_as(expansion_trajs)
        loss_shift_reg = (
            time_warp["all_donor_shifts"] - base_shift
        ).square().mean()
        donor_entropy = -(
            time_warp["donor_probability"]
            * torch.log(
                time_warp["donor_probability"].clamp_min(self.eps)
            )
        ).sum(dim=-1).mean()

        total = (
            float(cfg.get("LOSS_WEIGHT_DONOR", 1.0)) * loss_donor
            + float(cfg.get("LOSS_WEIGHT_GEOMETRY", 0.25)) * loss_geometry
            + float(cfg.get("LOSS_WEIGHT_HORIZON", 1.0)) * loss_horizon
            + float(cfg.get("LOSS_WEIGHT_GAIN", 0.25)) * loss_gain
            + float(cfg.get("LOSS_WEIGHT_ADMISSION", 1.0))
            * loss_admission
            + float(cfg.get("LOSS_WEIGHT_SELECTION_LISTWISE", 1.0))
            * loss_selection_listwise
            + float(cfg.get("LOSS_WEIGHT_CONFIDENCE_LISTWISE", 1.0))
            * loss_confidence_listwise
            + float(cfg.get("LOSS_WEIGHT_PAIRWISE", 0.5)) * loss_pairwise
            + float(cfg.get("LOSS_WEIGHT_GLOBAL_AP", 0.25))
            * loss_global_ap
            + float(cfg.get("LOSS_WEIGHT_SHIFT_REG", 0.01))
            * loss_shift_reg
            + float(cfg.get("LOSS_WEIGHT_DONOR_ENTROPY", 0.001))
            * donor_entropy
        )

        with torch.no_grad():
            donor_target = donor_target_utility.argmax(dim=-1)
            donor_accuracy = (
                time_warp["donor_rank"] == donor_target
            ).type_as(expansion_trajs).mean()
            base_coverage = base_metrics["horizon_match"].any(
                dim=1
            ).type_as(expansion_trajs)
            all_coverage = all_metrics["horizon_match"].any(
                dim=1
            ).type_as(expansion_trajs)
            selected_expansion_rate = (
                torch.sigmoid(admission_logits)
                >= self.expansion_admission_threshold
            ).type_as(expansion_trajs).mean()

        return total, {
            "loss_interaction_time_warp": total,
            "loss_time_warp_donor": loss_donor,
            "loss_time_warp_geometry": loss_geometry,
            "loss_time_warp_horizon": loss_horizon,
            "loss_time_warp_gain": loss_gain,
            "loss_time_warp_admission": loss_admission,
            "loss_time_warp_selection_listwise": (
                loss_selection_listwise
            ),
            "loss_time_warp_confidence_listwise": (
                loss_confidence_listwise
            ),
            "loss_time_warp_pairwise": loss_pairwise,
            "loss_time_warp_global_ap": loss_global_ap,
            "loss_time_warp_shift_reg": loss_shift_reg,
            "time_warp_donor_entropy": donor_entropy,
            "time_warp_donor_accuracy": donor_accuracy,
            "time_warp_admission_target_rate": admission_target.mean(),
            "time_warp_admission_probability": torch.sigmoid(
                admission_logits
            ).mean(),
            "time_warp_admitted_rate": selected_expansion_rate,
            "time_warp_rescue_target_rate": (
                expansion_rescue_utility > 0.0
            ).type_as(expansion_trajs).mean(),
            "time_warp_base_coverage_3s": base_coverage[:, 0].mean(),
            "time_warp_base_coverage_5s": base_coverage[:, 1].mean(),
            "time_warp_base_coverage_8s": base_coverage[:, 2].mean(),
            "time_warp_all_coverage_3s": all_coverage[:, 0].mean(),
            "time_warp_all_coverage_5s": all_coverage[:, 1].mean(),
            "time_warp_all_coverage_8s": all_coverage[:, 2].mean(),
            "time_warp_oracle_gain": (
                all_coverage - base_coverage
            ).clamp_min(0.0).mean(),
            "time_warp_shift_abs_seconds": time_warp[
                "selected_shifts"
            ].abs().mean(),
            "time_warp_confidence_residual_abs": time_warp[
                "confidence_residual"
            ].abs().mean(),
            "time_warp_selection_residual_abs": time_warp[
                "selection_residual"
            ].abs().mean(),
            "joint_oracle_ade": all_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": all_metrics["fde"].amin(dim=1).mean(),
        }

    def _interaction_set_replacement_loss(
        self, final_output, input_dict, reciprocal=False
    ):
        if reciprocal:
            candidate_source = final_output.get("reciprocal_augmentation")
            replacement_module = self.reciprocal_set_selector
            cfg = self.reciprocal_set_selector_cfg
        else:
            candidate_source = final_output.get(
                "interaction_time_warp_bank"
            )
            module = self.interaction_time_warp_bank
            replacement_module = (
                module.replacement_selector if module is not None else None
            )
            cfg = (
                module.replacement_selector_cfg
                if module is not None
                else {}
            )
        if candidate_source is None:
            raise RuntimeError(
                "Final output is missing relative replacement candidates"
            )
        selector = candidate_source.get("set_marginal_replacement")
        if selector is None or replacement_module is None:
            raise RuntimeError(
                "Final output is missing set-marginal replacement actions"
            )

        base_trajs = final_output["protected_base_joint_trajs"].detach()
        base_logits = final_output["protected_base_joint_logits"].detach()
        expansion_trajs = candidate_source["trajectories"].detach()
        base_metrics = self._official_match_quality(base_trajs, input_dict)
        expansion_metrics = self._official_match_quality(
            expansion_trajs, input_dict
        )
        targets = build_confidence_preserving_swap_targets(
            base_match=base_metrics["horizon_match"],
            expansion_match=expansion_metrics["horizon_match"],
            base_cost=base_metrics["horizon_cost"].detach(),
            expansion_cost=expansion_metrics["horizon_cost"].detach(),
            base_probability=torch.softmax(base_logits, dim=-1),
            pair_valid=base_metrics["pair_valid"],
            horizon_weights=replacement_module.horizon_weights,
            min_credit_gain=float(cfg.get("MIN_CREDIT_GAIN", 1e-4)),
            geometry_tie_weight=float(
                cfg.get("GEOMETRY_TIE_WEIGHT", 0.02)
            ),
            target_temperature=float(
                cfg.get("TARGET_TEMPERATURE", 0.03)
            ),
            near_best_tolerance=float(
                cfg.get("NEAR_BEST_TOLERANCE", 0.01)
            ),
        )

        if replacement_module.aligned_actions_only:
            donor_indices = candidate_source["donor_indices"].long()

            def gather_aligned(value):
                if value.ndim == 3:
                    return torch.gather(
                        value, 2, donor_indices[:, :, None]
                    ).squeeze(2)
                if value.ndim == 4:
                    return torch.gather(
                        value,
                        2,
                        donor_indices[:, :, None, None].expand(
                            -1, -1, 1, value.shape[-1]
                        ),
                    ).squeeze(2)
                raise ValueError(
                    f"Unsupported aligned target rank {value.ndim}"
                )

            for key in (
                "positive_action",
                "valid_action",
                "credit_gain",
                "geometry_gain",
                "action_gain",
                "horizon_credit_gain",
                "horizon_geometry_gain",
                "action_credit",
                "action_coverage",
            ):
                targets[key] = gather_aligned(targets[key])

            positive_action = targets["positive_action"]
            action_gain = targets["action_gain"]
            masked_gain = action_gain.masked_fill(
                ~positive_action, -torch.inf
            )
            best_gain, best_action = masked_gain.max(dim=-1)
            positive_scene = torch.isfinite(best_gain)
            near_best = positive_action & (
                action_gain
                >= best_gain[:, None]
                - float(cfg.get("NEAR_BEST_TOLERANCE", 0.01))
            )
            target_temperature = max(
                float(cfg.get("TARGET_TEMPERATURE", 0.03)), 1e-4
            )
            action_target = torch.softmax(
                (action_gain / target_temperature).masked_fill(
                    ~near_best, -1e4
                ),
                dim=-1,
            )
            action_target = (
                action_target
                * positive_scene[:, None].type_as(action_target)
            )
            targets["target_distribution"] = torch.cat(
                [
                    (~positive_scene).type_as(action_target)[:, None],
                    action_target,
                ],
                dim=-1,
            )
            targets["hard_target"] = torch.where(
                positive_scene,
                best_action + 1,
                torch.zeros_like(best_action),
            )
            targets["positive_scene"] = positive_scene

        action_logits = selector["action_logits"]
        target_distribution = targets["target_distribution"].detach()
        positive_scene = targets["positive_scene"]
        decision_conditional_logits = selector[
            "decision_conditional_logits"
        ]
        decision_pair_action_logits = selector["pair_action_logits"].flatten(1)
        decision_horizon_prediction = selector[
            "decision_horizon_gain"
        ]
        decision_gain_prediction = selector[
            "decision_predicted_gain"
        ]
        decision_gain_scale = selector.get(
            "decision_predicted_gain_scale"
        )
        decision_utility_lcb = selector.get("decision_utility_lcb")
        decision_policy_logits = selector.get(
            "decision_policy_logits", decision_conditional_logits
        )
        target_positive_action = targets["positive_action"].flatten(1)
        target_valid_action = targets["valid_action"].flatten(1)
        target_action_gain = targets["action_gain"].flatten(1)
        target_credit_gain = targets["credit_gain"].flatten(1)
        target_horizon_gain = targets["horizon_credit_gain"].reshape(
            targets["horizon_credit_gain"].shape[0],
            -1,
            targets["horizon_credit_gain"].shape[-1],
        )
        if not replacement_module.aligned_actions_only:
            decision_conditional_logits = decision_conditional_logits.flatten(1)
            decision_gain_prediction = decision_gain_prediction.flatten(1)
            if decision_gain_scale is not None:
                decision_gain_scale = decision_gain_scale.flatten(1)
            if decision_utility_lcb is not None:
                decision_utility_lcb = decision_utility_lcb.flatten(1)
        dtype = action_logits.dtype
        device = action_logits.device
        sample_weight = self._pair_sample_weights(input_dict, device, dtype)
        positive_scene_weight = float(
            cfg.get("POSITIVE_SCENE_WEIGHT", 2.0)
        )
        action_sample_weight = sample_weight * (
            1.0
            + (positive_scene_weight - 1.0)
            * positive_scene.type_as(sample_weight)
        )

        dense_marginal_action = bool(
            cfg.get("DENSE_MARGINAL_ACTION", False)
        )
        if dense_marginal_action:
            action_target = target_positive_action.type_as(
                decision_pair_action_logits
            )
            action_bce = F.binary_cross_entropy_with_logits(
                decision_pair_action_logits,
                action_target,
                reduction="none",
            )
            action_focal_gamma = float(
                cfg.get("ACTION_FOCAL_GAMMA", 0.0)
            )
            if action_focal_gamma > 0.0:
                action_probability = torch.sigmoid(
                    decision_pair_action_logits
                )
                action_target_probability = torch.where(
                    action_target > 0.5,
                    action_probability,
                    1.0 - action_probability,
                )
                action_bce = action_bce * (
                    1.0 - action_target_probability
                ).clamp_min(self.eps).pow(action_focal_gamma)
            action_positive_weight = float(
                cfg.get("ACTION_POSITIVE_WEIGHT", 4.0)
            )
            action_weight = 1.0 + (
                action_positive_weight - 1.0
            ) * action_target
            action_per_sample = (
                action_bce * action_weight
            ).sum(dim=1) / action_weight.sum(dim=1).clamp_min(
                1.0
            )
            loss_action = self._weighted_batch_mean(
                action_per_sample, action_sample_weight
            )
        else:
            action_ce = -(
                target_distribution
                * F.log_softmax(action_logits, dim=-1)
            ).sum(dim=-1)
            loss_action = self._weighted_batch_mean(
                action_ce, action_sample_weight
            )

        change_target = positive_scene.type_as(selector["change_logit"])
        change_bce = F.binary_cross_entropy_with_logits(
            selector["change_logit"], change_target, reduction="none"
        )
        change_positive_weight = float(
            cfg.get("CHANGE_POSITIVE_WEIGHT", 3.0)
        )
        change_weight = 1.0 + (
            change_positive_weight - 1.0
        ) * change_target
        loss_change = self._weighted_batch_mean(
            change_bce * change_weight, sample_weight
        )

        pair_target = target_distribution[:, 1:]
        pair_target_sum = pair_target.sum(dim=-1, keepdim=True)
        normalized_pair_target = pair_target / pair_target_sum.clamp_min(
            self.eps
        )
        conditional_ce = -(
            normalized_pair_target
            * F.log_softmax(
                decision_conditional_logits, dim=-1
            )
        ).sum(dim=-1)
        loss_conditional = self._weighted_batch_mean(
            conditional_ce,
            sample_weight * positive_scene.type_as(sample_weight),
        )

        risk_calibrated_utility = bool(
            cfg.get("RISK_CALIBRATED_UTILITY", False)
        )
        conditional_flat = (
            decision_utility_lcb
            if risk_calibrated_utility
            else decision_policy_logits
        )
        valid_action_flat = target_valid_action
        raw_action_gain_target = (
            target_credit_gain.detach()
            if risk_calibrated_utility
            else target_action_gain.detach()
        ).clamp(
            -replacement_module.max_predicted_gain,
            replacement_module.max_predicted_gain,
        )
        if risk_calibrated_utility:
            action_gain_target = build_safe_swap_utility_targets(
                raw_action_gain_target,
                target_valid_action,
                max_abs_gain=replacement_module.max_predicted_gain,
                invalid_penalty=float(
                    cfg.get("UTILITY_INVALID_ACTION_PENALTY", 0.01)
                ),
            )
        else:
            action_gain_target = raw_action_gain_target
        action_gain_flat = action_gain_target
        conditional_temperature = max(
            float(cfg.get("CONDITIONAL_TEMPERATURE", 0.35)), 1e-3
        )
        if risk_calibrated_utility:
            no_op = conditional_flat.new_zeros(
                conditional_flat.shape[0], 1
            )
            regret_logits = torch.cat([no_op, conditional_flat], dim=-1)
            regret_target = torch.cat([no_op, action_gain_flat], dim=-1)
            conditional_probability = torch.softmax(
                regret_logits / conditional_temperature, dim=-1
            )
            expected_action_gain = (
                conditional_probability * regret_target
            ).sum(dim=-1)
            oracle_action_gain = regret_target.amax(dim=-1)
        else:
            conditional_probability = torch.softmax(
                conditional_flat.masked_fill(~valid_action_flat, -1e4)
                / conditional_temperature,
                dim=-1,
            )
            expected_action_gain = (
                conditional_probability * action_gain_flat
            ).sum(dim=-1)
            oracle_action_gain = action_gain_flat.masked_fill(
                ~valid_action_flat, -torch.inf
            ).amax(dim=-1)
            oracle_action_gain = torch.where(
                torch.isfinite(oracle_action_gain),
                oracle_action_gain,
                torch.zeros_like(oracle_action_gain),
            )
        action_regret = (
            oracle_action_gain - expected_action_gain
        ).clamp_min(0.0)
        loss_expected_regret = self._weighted_batch_mean(
            action_regret,
            sample_weight * positive_scene.type_as(sample_weight),
        )

        positive_action_flat = target_positive_action
        # Invalid swaps are hard negatives at deployment: unlike the training
        # target builder, inference has no GT validity mask available. They must
        # therefore be ranked below valid positive actions, not omitted.
        negative_action_flat = ~positive_action_flat
        ranking_temperature = max(
            float(cfg.get("ACTION_RANKING_TEMPERATURE", 0.20)), 1e-3
        )
        positive_score = ranking_temperature * torch.logsumexp(
            conditional_flat.masked_fill(
                ~positive_action_flat, -1e4
            )
            / ranking_temperature,
            dim=-1,
        )
        negative_score = ranking_temperature * torch.logsumexp(
            conditional_flat.masked_fill(
                ~negative_action_flat, -1e4
            )
            / ranking_temperature,
            dim=-1,
        )
        has_negative = negative_action_flat.any(dim=-1)
        negative_score = torch.where(
            has_negative,
            negative_score,
            positive_score.detach()
            - float(cfg.get("ACTION_RANKING_MARGIN", 0.50)),
        )
        ranking_margin = float(cfg.get("ACTION_RANKING_MARGIN", 0.50))
        action_ranking_error = ranking_temperature * F.softplus(
            (
                ranking_margin
                - positive_score
                + negative_score
            )
            / ranking_temperature
        )
        loss_action_ranking = self._weighted_batch_mean(
            action_ranking_error,
            sample_weight * positive_scene.type_as(sample_weight),
        )

        horizon_target = target_horizon_gain.detach()
        horizon_prediction = decision_horizon_prediction
        horizon_valid = base_metrics["pair_valid"][:, None].expand_as(
            horizon_target
        )
        horizon_error = F.smooth_l1_loss(
            horizon_prediction,
            horizon_target,
            reduction="none",
            beta=float(cfg.get("HORIZON_GAIN_BETA", 0.05)),
        )
        nonzero_horizon_weight = float(
            cfg.get("NONZERO_HORIZON_WEIGHT", 3.0)
        )
        horizon_loss_weight = horizon_valid.type_as(horizon_error) * (
            1.0
            + (nonzero_horizon_weight - 1.0)
            * (horizon_target.abs() > 1e-6).type_as(horizon_error)
        )
        horizon_sample = (
            horizon_error * horizon_loss_weight
        ).sum(dim=(1, 2)) / horizon_loss_weight.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss_horizon_gain = self._weighted_batch_mean(
            horizon_sample, sample_weight
        )

        # Risk-calibrated deployment predicts the confidence-weighted official
        # credit change itself. Geometry remains an auxiliary tie-breaker but
        # must not turn a zero-credit action into an accepted replacement.
        if risk_calibrated_utility:
            if decision_gain_scale is None:
                raise RuntimeError(
                    "Risk-calibrated utility requires predicted gain scale"
                )
            # Dividing by the fixed scale floor only removes a constant from
            # the Laplace NLL. Gradients are unchanged while the reported loss
            # remains non-negative and easier to audit for instability.
            gain_error = (
                (decision_gain_prediction - action_gain_target).abs()
                / decision_gain_scale.clamp_min(self.eps)
                + (
                    decision_gain_scale.clamp_min(self.eps)
                    / replacement_module.utility_scale_floor
                ).log()
            )
        else:
            action_gain_target = raw_action_gain_target
            gain_error = F.smooth_l1_loss(
                decision_gain_prediction,
                action_gain_target,
                reduction="none",
                beta=float(cfg.get("ACTION_GAIN_BETA", 0.05)),
            )
        gain_weight = 1.0 + (
            float(cfg.get("POSITIVE_ACTION_WEIGHT", 3.0)) - 1.0
        ) * (
            target_valid_action & (target_credit_gain > 0.0)
        ).type_as(gain_error)
        gain_sample = (gain_error * gain_weight).mean(dim=1)
        loss_gain = self._weighted_batch_mean(gain_sample, sample_weight)

        if risk_calibrated_utility:
            group_ids = official_ap_group_ids(
                input_dict,
                decision_utility_lcb.shape[0],
                decision_utility_lcb.device,
            )
            loss_bucket_action_ranking = (
                self._bucket_action_utility_ranking_loss(
                    decision_utility_lcb,
                    action_gain_target,
                    group_ids,
                    cfg,
                )
            )
        else:
            loss_bucket_action_ranking = action_logits.sum() * 0.0

        if risk_calibrated_utility:
            safe_action_utility = action_gain_target
            no_op_utility = torch.zeros_like(
                safe_action_utility[:, :1]
            )
            deployment_utility = torch.cat(
                [no_op_utility, safe_action_utility], dim=-1
            )
            policy_temperature = max(
                float(cfg.get("UTILITY_POLICY_TEMPERATURE", 0.03)),
                1e-3,
            )
            deployment_probability = torch.softmax(
                action_logits / policy_temperature, dim=-1
            )
            expected_deployment_utility = (
                deployment_probability * deployment_utility
            ).sum(dim=-1)
            oracle_deployment_utility = deployment_utility.amax(dim=-1)
            deployment_regret = (
                oracle_deployment_utility
                - expected_deployment_utility
            )
            loss_deployment_utility = self._weighted_batch_mean(
                deployment_regret, sample_weight
            )
        else:
            safe_action_utility = target_action_gain
            expected_deployment_utility = torch.zeros_like(sample_weight)
            oracle_deployment_utility = torch.zeros_like(sample_weight)
            deployment_regret = torch.zeros_like(sample_weight)
            loss_deployment_utility = action_logits.sum() * 0.0

        false_accept_margin = float(
            cfg.get("FALSE_ACCEPT_MARGIN", 0.5)
        )
        if replacement_module.hierarchical_deployment:
            false_accept = F.relu(
                selector["change_logit"]
                - replacement_module.gate_logit_threshold
                + false_accept_margin
            )
        else:
            no_op_logit = action_logits[:, 0]
            best_pair_logit = decision_pair_action_logits.amax(dim=-1)
            false_accept = F.relu(
                best_pair_logit - no_op_logit + false_accept_margin
            )
        loss_false_accept = self._weighted_batch_mean(
            false_accept,
            sample_weight * (~positive_scene).type_as(sample_weight),
        )

        if bool(cfg.get("USE_SET_MIL", False)):
            mil_temperature = float(cfg.get("SET_MIL_TEMPERATURE", 0.2))
            mil_margin = float(cfg.get("SET_MIL_MARGIN", 0.5))
            pair_logits = decision_pair_action_logits
            positive_action = target_positive_action
            positive_logits = pair_logits.masked_fill(
                ~positive_action, -1e4
            )
            best_positive_logit = mil_temperature * torch.logsumexp(
                positive_logits / mil_temperature,
                dim=-1,
            )
            best_action_logit = mil_temperature * torch.logsumexp(
                pair_logits / mil_temperature,
                dim=-1,
            )
            positive_mil = mil_temperature * F.softplus(
                (mil_margin - best_positive_logit) / mil_temperature
            )
            negative_mil = mil_temperature * F.softplus(
                (mil_margin + best_action_logit) / mil_temperature
            )
            mil_per_sample = torch.where(
                positive_scene, positive_mil, negative_mil
            )
            mil_positive_weight = float(
                cfg.get("SET_MIL_POSITIVE_WEIGHT", 1.5)
            )
            mil_sample_weight = sample_weight * (
                1.0
                + (mil_positive_weight - 1.0)
                * positive_scene.type_as(sample_weight)
            )
            loss_set_mil = self._weighted_batch_mean(
                mil_per_sample, mil_sample_weight
            )
        else:
            loss_set_mil = action_logits.sum() * 0.0
            best_positive_logit = conditional_flat.masked_fill(
                ~target_positive_action, -torch.inf
            ).amax(dim=-1)
            best_positive_logit = torch.where(
                torch.isfinite(best_positive_logit),
                best_positive_logit,
                torch.zeros_like(best_positive_logit),
            )
            best_action_logit = decision_pair_action_logits.amax(dim=-1)

        total = (
            float(cfg.get("LOSS_WEIGHT_ACTION", 1.0)) * loss_action
            + float(cfg.get("LOSS_WEIGHT_CHANGE", 1.0)) * loss_change
            + float(cfg.get("LOSS_WEIGHT_CONDITIONAL", 1.0))
            * loss_conditional
            + float(cfg.get("LOSS_WEIGHT_HORIZON_GAIN", 0.5))
            * loss_horizon_gain
            + float(cfg.get("LOSS_WEIGHT_ACTION_GAIN", 0.25)) * loss_gain
            + float(cfg.get("LOSS_WEIGHT_FALSE_ACCEPT", 0.5))
            * loss_false_accept
            + float(cfg.get("LOSS_WEIGHT_SET_MIL", 0.0)) * loss_set_mil
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_REGRET", 0.0))
            * loss_expected_regret
            + float(cfg.get("LOSS_WEIGHT_ACTION_RANKING", 0.0))
            * loss_action_ranking
            + float(cfg.get("LOSS_WEIGHT_DEPLOYMENT_UTILITY", 0.0))
            * loss_deployment_utility
            + float(cfg.get("LOSS_WEIGHT_BUCKET_ACTION_RANKING", 0.0))
            * loss_bucket_action_ranking
        )

        with torch.no_grad():
            hard_action = selector["hard_action"]
            predicted_change = hard_action > 0
            target_change = positive_scene
            true_positive = predicted_change & target_change
            precision = true_positive.sum().type_as(action_logits) / (
                predicted_change.sum().clamp_min(1)
            )
            recall = true_positive.sum().type_as(action_logits) / (
                target_change.sum().clamp_min(1)
            )
            positive_action = target_positive_action
            acceptance_threshold = (
                replacement_module.utility_acceptance_threshold
                if risk_calibrated_utility
                else replacement_module.acceptance_logit_threshold
            )
            predicted_positive_action = (
                decision_pair_action_logits > acceptance_threshold
            )
            action_true_positive = (
                predicted_positive_action & positive_action
            )
            action_precision = action_true_positive.sum().type_as(
                action_logits
            ) / predicted_positive_action.sum().clamp_min(1)
            action_recall = action_true_positive.sum().type_as(
                action_logits
            ) / positive_action.sum().clamp_min(1)
            flat_gain = (
                target_credit_gain
                if risk_calibrated_utility
                else target_action_gain
            )
            selected_index = (hard_action - 1).clamp_min(0)
            selected_gain = flat_gain.gather(
                1, selected_index[:, None]
            ).squeeze(1)
            selected_gain = torch.where(
                predicted_change,
                selected_gain,
                torch.zeros_like(selected_gain),
            )
            oracle_gain = flat_gain.masked_fill(
                ~target_valid_action, -torch.inf
            ).amax(dim=-1)
            oracle_gain = torch.where(
                torch.isfinite(oracle_gain),
                oracle_gain.clamp_min(0.0),
                torch.zeros_like(oracle_gain),
            )
            base_coverage = targets["base_coverage"].type_as(action_logits)
            pool_coverage = torch.cat(
                [
                    base_metrics["horizon_match"],
                    expansion_metrics["horizon_match"],
                ],
                dim=1,
            ).any(dim=1).type_as(action_logits)

        return total, {
            "loss_interaction_set_replacement": total,
            "loss_replacement_action": loss_action,
            "loss_replacement_change": loss_change,
            "loss_replacement_conditional": loss_conditional,
            "loss_replacement_horizon_gain": loss_horizon_gain,
            "loss_replacement_action_gain": loss_gain,
            "loss_replacement_false_accept": loss_false_accept,
            "loss_replacement_set_mil": loss_set_mil,
            "loss_replacement_expected_regret": loss_expected_regret,
            "loss_replacement_action_ranking": loss_action_ranking,
            "loss_replacement_deployment_utility": (
                loss_deployment_utility
            ),
            "loss_replacement_bucket_action_ranking": (
                loss_bucket_action_ranking
            ),
            "replacement_target_rate": target_change.float().mean(),
            "replacement_predicted_rate": predicted_change.float().mean(),
            "replacement_action_accuracy": (
                hard_action == targets["hard_target"]
            ).float().mean(),
            "replacement_change_precision": precision,
            "replacement_change_recall": recall,
            "replacement_positive_action_rate": positive_action.float().mean(),
            "replacement_predicted_action_rate": (
                predicted_positive_action.float().mean()
            ),
            "replacement_dense_action_precision": action_precision,
            "replacement_dense_action_recall": action_recall,
            "replacement_best_action_margin": (
                decision_pair_action_logits.amax(dim=-1)
                - acceptance_threshold
            ).mean(),
            "replacement_positive_best_margin": torch.where(
                positive_scene,
                best_positive_logit
                - acceptance_threshold,
                torch.zeros_like(best_positive_logit),
            ).sum()
            / positive_scene.sum().clamp_min(1),
            "replacement_negative_best_margin": torch.where(
                ~positive_scene,
                best_action_logit
                - acceptance_threshold,
                torch.zeros_like(best_action_logit),
            ).sum()
            / (~positive_scene).sum().clamp_min(1),
            "replacement_oracle_gain": oracle_gain.mean(),
            "replacement_selected_gain": selected_gain.mean(),
            "replacement_predicted_gain_mean": (
                decision_gain_prediction.mean()
            ),
            "replacement_predicted_scale_mean": (
                decision_gain_scale.mean()
                if decision_gain_scale is not None
                else action_logits.sum() * 0.0
            ),
            "replacement_utility_lcb_mean": (
                decision_utility_lcb.mean()
                if decision_utility_lcb is not None
                else action_logits.sum() * 0.0
            ),
            "replacement_expected_deployment_utility": (
                expected_deployment_utility.mean()
            ),
            "replacement_oracle_deployment_utility": (
                oracle_deployment_utility.mean()
            ),
            "replacement_deployment_regret": deployment_regret.mean(),
            "replacement_expected_action_gain": (
                expected_action_gain.mean()
            ),
            "replacement_expected_action_regret": action_regret.mean(),
            "replacement_base_coverage_3s": base_coverage[:, 0].mean(),
            "replacement_base_coverage_5s": base_coverage[:, 1].mean(),
            "replacement_base_coverage_8s": base_coverage[:, 2].mean(),
            "replacement_pool_coverage_3s": pool_coverage[:, 0].mean(),
            "replacement_pool_coverage_5s": pool_coverage[:, 1].mean(),
            "replacement_pool_coverage_8s": pool_coverage[:, 2].mean(),
            "joint_oracle_ade": torch.cat(
                [base_metrics["ade"], expansion_metrics["ade"]], dim=1
            ).amin(dim=1).mean(),
            "joint_oracle_fde": torch.cat(
                [base_metrics["fde"], expansion_metrics["fde"]], dim=1
            ).amin(dim=1).mean(),
        }

    def _reciprocal_world_set_decoder_loss(self, final_output, input_dict):
        decoded = final_output.get("reciprocal_world_set_decoder")
        module = self.reciprocal_world_set_decoder
        if decoded is None or module is None:
            raise RuntimeError("Missing reciprocal world set decoder output")
        cfg = self.reciprocal_world_set_decoder_cfg
        pool = decoded["candidate_trajectories"]
        final_trajs = decoded["final_trajectories"]
        anchor_trajs = decoded["anchor_trajectories"]
        final_logits = decoded["final_logits"]
        sample_weight = self._pair_sample_weights(
            input_dict, final_trajs.device, final_trajs.dtype
        )
        horizon_weight = module.horizon_weights.type_as(final_trajs)

        pool_metrics = self._official_match_quality(pool, input_dict)
        final_metrics = self._official_match_quality(final_trajs, input_dict)
        anchor_metrics = self._official_match_quality(anchor_trajs, input_dict)
        valid = final_metrics["pair_valid"].type_as(final_trajs)
        valid_horizon_weight = valid * horizon_weight[None]
        valid_horizon_denominator = valid_horizon_weight.sum(
            dim=-1
        ).clamp_min(self.eps)

        pool_credit = build_soft_map_credit_targets(
            horizon_match=pool_metrics["horizon_match"],
            horizon_cost=pool_metrics["horizon_cost"].detach(),
            pair_valid=pool_metrics["pair_valid"],
        )
        boundary_temperature = max(
            float(cfg.get("BOUNDARY_TEMPERATURE", 0.20)), self.eps
        )
        pool_boundary = torch.sigmoid(
            (1.0 - pool_metrics["horizon_cost"].detach())
            / boundary_temperature
        )
        pool_horizon_utility = (
            float(cfg.get("UTILITY_CREDIT_WEIGHT", 2.0))
            * pool_credit["credited_target"].type_as(final_trajs)
            + float(cfg.get("UTILITY_MATCH_WEIGHT", 0.5))
            * pool_metrics["horizon_match"].type_as(final_trajs)
            + float(cfg.get("UTILITY_BOUNDARY_WEIGHT", 1.0)) * pool_boundary
            - float(cfg.get("UTILITY_COST_WEIGHT", 0.25))
            * pool_metrics["horizon_cost"].detach().clamp_max(4.0)
        )
        pool_utility = (
            pool_horizon_utility
            * valid_horizon_weight[:, None]
        ).sum(dim=-1) / valid_horizon_denominator[:, None]

        with torch.no_grad():
            signature = pool.index_select(3, module.measurement_steps).flatten(2)
            candidate_distance = torch.cdist(
                signature.float(), signature.float()
            ).type_as(pool)
            target_indices = torch.empty(
                pool.shape[0],
                self.num_output_modes,
                dtype=torch.long,
                device=pool.device,
            )
            target_diversity_weight = float(
                cfg.get("TARGET_DIVERSITY_WEIGHT", 0.20)
            )
            target_diversity_scale = max(
                float(cfg.get("TARGET_DIVERSITY_SCALE", 3.0)), self.eps
            )
            base_logits = final_output["protected_base_joint_logits"]
            for batch_idx in range(pool.shape[0]):
                available = torch.ones(
                    pool.shape[1], dtype=torch.bool, device=pool.device
                )
                selected = []
                for _ in range(self.num_output_modes):
                    score = pool_utility[batch_idx].clone()
                    if selected:
                        novelty = candidate_distance[
                            batch_idx, :, selected
                        ].amin(dim=-1)
                        score = score + target_diversity_weight * (
                            novelty / target_diversity_scale
                        ).clamp(max=1.0)
                    score = score.masked_fill(~available, -torch.inf)
                    candidate_idx = int(score.argmax().item())
                    selected.append(candidate_idx)
                    available[candidate_idx] = False
                slot_order = base_logits[batch_idx].argsort(descending=True)
                target_indices[batch_idx, slot_order] = torch.tensor(
                    selected, device=pool.device, dtype=torch.long
                )

        assignment_temperature = max(
            float(cfg.get("ASSIGNMENT_LOSS_TEMPERATURE", 0.7)), self.eps
        )
        assignment_ce = F.cross_entropy(
            (
                decoded["assignment_logits"] / assignment_temperature
            ).reshape(-1, pool.shape[1]),
            target_indices.reshape(-1),
            reduction="none",
        ).reshape(pool.shape[0], self.num_output_modes)
        slot_weight = torch.softmax(
            final_output["protected_base_joint_logits"].detach(), dim=-1
        )
        loss_assignment = self._weighted_batch_mean(
            (
                assignment_ce
                * (0.5 + self.num_output_modes * slot_weight)
            ).mean(dim=-1),
            sample_weight,
        )

        expected_utility = torch.einsum(
            "bsc,bc->bs", decoded["soft_assignment"], pool_utility.detach()
        )
        loss_expected_utility = self._weighted_batch_mean(
            -(expected_utility * slot_weight).sum(dim=-1), sample_weight
        )
        inclusion_probability = 1.0 - (
            1.0 - decoded["soft_assignment"].clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        soft_pool_success = 1.0 - (
            1.0
            - inclusion_probability[:, :, None]
            * pool_boundary.clamp(max=1.0 - 1e-4)
        ).prod(dim=1)
        coverage_per_sample = -(
            soft_pool_success.clamp_min(self.eps).log()
            * valid_horizon_weight
        ).sum(dim=-1) / valid_horizon_denominator
        loss_selection_coverage = self._weighted_batch_mean(
            coverage_per_sample, sample_weight
        )

        normalized_assignment = F.normalize(
            decoded["soft_assignment"], p=2, dim=-1
        )
        assignment_similarity = torch.matmul(
            normalized_assignment, normalized_assignment.transpose(1, 2)
        )
        off_diagonal = ~torch.eye(
            self.num_output_modes,
            device=pool.device,
            dtype=torch.bool,
        )[None]
        loss_uniqueness = assignment_similarity.masked_select(
            off_diagonal.expand_as(assignment_similarity)
        ).mean()

        final_credit = build_soft_map_credit_targets(
            horizon_match=final_metrics["horizon_match"],
            horizon_cost=final_metrics["horizon_cost"].detach(),
            pair_valid=final_metrics["pair_valid"],
        )
        fallback_temperature = max(
            float(cfg.get("CONFIDENCE_FALLBACK_TEMPERATURE", 0.25)), self.eps
        )
        fallback_target = torch.softmax(
            -final_metrics["horizon_cost"].detach() / fallback_temperature,
            dim=1,
        )
        credited_target = final_credit["credited_target"].type_as(final_logits)
        confidence_target = torch.where(
            final_credit["has_match"][:, None],
            credited_target,
            fallback_target,
        )
        confidence_temperature = max(
            float(cfg.get("CONFIDENCE_LOSS_TEMPERATURE", 0.7)), self.eps
        )
        confidence_log_probability = F.log_softmax(
            final_logits / confidence_temperature, dim=1
        )[:, :, None]
        confidence_horizon_ce = -(
            confidence_target * confidence_log_probability
        ).sum(dim=1)
        confidence_per_sample = (
            confidence_horizon_ce * valid_horizon_weight
        ).sum(dim=-1) / valid_horizon_denominator
        loss_confidence = self._weighted_batch_mean(
            confidence_per_sample, sample_weight
        )
        final_probability = torch.softmax(final_logits, dim=-1)
        group_ids = official_ap_group_ids(
            input_dict, final_logits.shape[0], final_logits.device
        )
        loss_global_ap = self._global_ap_surrogate(
            final_probability, final_credit, group_ids, cfg
        )

        gt = final_metrics["gt"]
        mask = final_metrics["mask"].type_as(final_trajs)
        point_error = F.smooth_l1_loss(
            final_trajs,
            gt[:, None].expand_as(final_trajs),
            reduction="none",
            beta=float(cfg.get("REGRESSION_BETA", 0.5)),
        ).sum(dim=-1)
        mode_regression = (
            point_error * mask[:, None]
        ).sum(dim=(-1, -2)) / mask[:, None].sum(
            dim=(-1, -2)
        ).clamp_min(1.0)
        responsibility_temperature = max(
            float(cfg.get("RESPONSIBILITY_TEMPERATURE", 0.20)), self.eps
        )
        responsibility = torch.softmax(
            -final_metrics["quality"].detach()
            / responsibility_temperature,
            dim=-1,
        )
        loss_regression = self._weighted_batch_mean(
            (mode_regression * responsibility).sum(dim=-1), sample_weight
        )

        softmin_temperature = max(
            float(cfg.get("SOFTMIN_TEMPERATURE", 0.20)), self.eps
        )

        def normalized_softmin(cost):
            return -softmin_temperature * (
                torch.logsumexp(-cost / softmin_temperature, dim=1)
                - math.log(cost.shape[1])
            )

        final_softmin = normalized_softmin(final_metrics["horizon_cost"])
        anchor_softmin = normalized_softmin(
            anchor_metrics["horizon_cost"].detach()
        )
        geometry_per_sample = (
            final_softmin * valid_horizon_weight
        ).sum(dim=-1) / valid_horizon_denominator
        loss_geometry = self._weighted_batch_mean(
            geometry_per_sample, sample_weight
        )
        no_harm = F.relu(
            final_softmin
            - anchor_softmin
            - float(cfg.get("NO_HARM_TOLERANCE", 0.01))
        )
        no_harm_per_sample = (
            no_harm * valid_horizon_weight
        ).sum(dim=-1) / valid_horizon_denominator
        loss_no_harm = self._weighted_batch_mean(
            no_harm_per_sample, sample_weight
        )

        final_signature = final_trajs.index_select(
            3, module.measurement_steps
        ).flatten(2)
        final_distance = torch.cdist(
            final_signature.float(), final_signature.float()
        ).type_as(final_trajs)
        diversity_margin = float(cfg.get("DIVERSITY_MARGIN", 1.5))
        diversity_penalty = F.relu(diversity_margin - final_distance)
        upper_triangle = torch.triu(
            torch.ones_like(diversity_penalty, dtype=torch.bool), diagonal=1
        )
        loss_diversity = diversity_penalty.masked_select(
            upper_triangle
        ).mean()

        dense_delta = decoded["dense_delta"]
        loss_residual_reg = dense_delta.square().mean()
        second_difference = (
            dense_delta[..., 2:, :]
            - 2.0 * dense_delta[..., 1:-1, :]
            + dense_delta[..., :-2, :]
        )
        loss_residual_smooth = second_difference.square().mean()
        loss_confidence_reg = decoded["confidence_residual"].square().mean()
        loss_assignment_reg = decoded["assignment_residual"].square().mean()

        total = (
            float(cfg.get("LOSS_WEIGHT_ASSIGNMENT", 1.0)) * loss_assignment
            + float(cfg.get("LOSS_WEIGHT_EXPECTED_UTILITY", 0.5))
            * loss_expected_utility
            + float(cfg.get("LOSS_WEIGHT_SELECTION_COVERAGE", 0.75))
            * loss_selection_coverage
            + float(cfg.get("LOSS_WEIGHT_UNIQUENESS", 0.10))
            * loss_uniqueness
            + float(cfg.get("LOSS_WEIGHT_CONFIDENCE", 1.0))
            * loss_confidence
            + float(cfg.get("LOSS_WEIGHT_GLOBAL_AP", 0.25))
            * loss_global_ap
            + float(cfg.get("LOSS_WEIGHT_REGRESSION", 1.0))
            * loss_regression
            + float(cfg.get("LOSS_WEIGHT_GEOMETRY", 0.5)) * loss_geometry
            + float(cfg.get("LOSS_WEIGHT_NO_HARM", 1.0)) * loss_no_harm
            + float(cfg.get("LOSS_WEIGHT_DIVERSITY", 0.05))
            * loss_diversity
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL_REG", 0.005))
            * loss_residual_reg
            + float(cfg.get("LOSS_WEIGHT_RESIDUAL_SMOOTH", 0.02))
            * loss_residual_smooth
            + float(cfg.get("LOSS_WEIGHT_CONFIDENCE_REG", 0.002))
            * loss_confidence_reg
            + float(cfg.get("LOSS_WEIGHT_ASSIGNMENT_REG", 0.001))
            * loss_assignment_reg
        )

        with torch.no_grad():
            identity = torch.arange(
                self.num_output_modes, device=pool.device
            )[None]
            selected_indices = decoded["selected_indices"]
            top_index = final_logits.argmax(dim=-1)
            batch_index = torch.arange(pool.shape[0], device=pool.device)
            top_match = final_metrics["horizon_match"][
                batch_index, top_index
            ].type_as(final_trajs)
            top_match = (
                top_match * valid_horizon_weight
            ).sum() / valid_horizon_weight.sum().clamp_min(1.0)
            pool_coverage = pool_metrics["horizon_match"].any(dim=1).float()
            base_coverage = pool_metrics["horizon_match"][
                :, : self.num_output_modes
            ].any(dim=1).float()

        return total, {
            "loss_reciprocal_world_set": total,
            "loss_world_set_assignment": loss_assignment,
            "loss_world_set_expected_utility": loss_expected_utility,
            "loss_world_set_selection_coverage": loss_selection_coverage,
            "loss_world_set_uniqueness": loss_uniqueness,
            "loss_world_set_confidence": loss_confidence,
            "loss_world_set_global_ap": loss_global_ap,
            "loss_world_set_regression": loss_regression,
            "loss_world_set_geometry": loss_geometry,
            "loss_world_set_no_harm": loss_no_harm,
            "loss_world_set_diversity": loss_diversity,
            "loss_world_set_residual_reg": loss_residual_reg,
            "loss_world_set_residual_smooth": loss_residual_smooth,
            "loss_world_set_confidence_reg": loss_confidence_reg,
            "loss_world_set_assignment_reg": loss_assignment_reg,
            "world_set_changed_scene_rate": (
                selected_indices != identity
            ).any(dim=-1).float().mean(),
            "world_set_expansion_rate": (
                selected_indices >= self.num_output_modes
            ).float().mean(),
            "world_set_oracle_expansion_rate": (
                target_indices >= self.num_output_modes
            ).float().mean(),
            "world_set_confidence_residual_abs": decoded[
                "confidence_residual"
            ].abs().mean(),
            "world_set_trajectory_delta_abs": dense_delta.abs().mean(),
            "world_set_top1_match": top_match,
            "world_set_base_coverage_3s": base_coverage[:, 0].mean(),
            "world_set_base_coverage_5s": base_coverage[:, 1].mean(),
            "world_set_base_coverage_8s": base_coverage[:, 2].mean(),
            "world_set_pool_coverage_3s": pool_coverage[:, 0].mean(),
            "world_set_pool_coverage_5s": pool_coverage[:, 1].mean(),
            "world_set_pool_coverage_8s": pool_coverage[:, 2].mean(),
            "world_set_anchor_oracle_ade": anchor_metrics["ade"].amin(
                dim=1
            ).mean(),
            "world_set_final_oracle_ade": final_metrics["ade"].amin(
                dim=1
            ).mean(),
            "world_set_anchor_oracle_fde": anchor_metrics["fde"].amin(
                dim=1
            ).mean(),
            "world_set_final_oracle_fde": final_metrics["fde"].amin(
                dim=1
            ).mean(),
            "joint_oracle_ade": final_metrics["ade"].amin(dim=1).mean(),
            "joint_oracle_fde": final_metrics["fde"].amin(dim=1).mean(),
        }

    def _base_generation_outputs(self, outputs):
        """Return decoder outputs before post-decoder candidate expansion.

        Protected expansion changes the final candidate axis from six base
        generation modes to a larger deployment pool.  The MTR prediction
        loss and structured-world loss must supervise the same six modes that
        produced ``loss_agent_pred_trajs`` and ``loss_agent_pred_scores``.
        """
        base_outputs = list(outputs)
        final_output = dict(outputs[-1])
        base_trajs = final_output.get("protected_base_joint_trajs")
        base_logits = final_output.get("protected_base_joint_logits")
        if (base_trajs is None) != (base_logits is None):
            raise RuntimeError(
                "Protected expansion must preserve both base trajectories "
                "and base logits"
            )
        if base_trajs is None:
            base_trajs = final_output["joint_trajs"]
            base_logits = final_output["joint_logits"]
        if (
            base_trajs.shape[1] != self.num_output_modes
            or base_logits.shape[1] != self.num_output_modes
        ):
            raise RuntimeError(
                "Base generation supervision requires exactly "
                f"{self.num_output_modes} modes, got trajectories="
                f"{base_trajs.shape[1]} and logits={base_logits.shape[1]}"
            )
        final_output["joint_trajs"] = base_trajs
        final_output["joint_logits"] = base_logits
        base_outputs[-1] = final_output
        return base_outputs

    def get_loss(self, state, outputs, input_dict):
        if self.train_stage in {"cwrr_response_r1", "cwrr_response_r2"}:
            final_output = outputs[-1]
            response = final_output.get("cwrr_response")
            if self.cwrr is None or response is None:
                raise RuntimeError("Missing CWRR response output during training")
            loss, metrics = self.cwrr.get_response_loss(
                response_output=response,
                world_hidden=final_output[
                    "protected_expansion_world_hidden"
                ].detach(),
                scene_context=state["scene_token"].detach(),
                context=state["cwrr_context"],
                input_dict=input_dict,
                official_quality_fn=self._official_match_quality,
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            tb.setdefault("joint_oracle_ade", 0.0)
            tb.setdefault("joint_oracle_fde", 0.0)
            return loss, tb
        if self.train_stage in {
            "cwrr_refine_r0",
            "cwrr_refine_r1",
            "cwrr_refine_r2",
        }:
            final_output = outputs[-1]
            refinement = final_output.get("cwrr_refinement")
            if self.cwrr is None or refinement is None:
                raise RuntimeError(
                    "Missing CWRR refinement output during training"
                )
            loss, metrics = self.cwrr.refinement_loss(
                refinement, input_dict, self._official_match_quality
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage in {
            "candidate_bank_sequential_mode_warmup",
            "candidate_bank_sequential_mode_refine",
            "candidate_bank_sequential_mode_joint",
        }:
            loss, metrics = self._sequential_mode_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_improvement_gated_flow":
            loss, metrics = self._improvement_gated_residual_flow_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage in {
            "candidate_bank_residual_flow_geometry",
            "candidate_bank_residual_flow_score",
            "candidate_bank_residual_flow_warmup",
            "candidate_bank_residual_flow_joint",
            "candidate_bank_residual_flow_generation_joint",
            "candidate_bank_residual_flow_full_convergence",
        }:
            if self.candidate_residual_flow is None:
                raise RuntimeError(
                    "Candidate residual-flow stage requires "
                    "CANDIDATE_RESIDUAL_FLOW.ENABLED=True"
                )
            final_output = outputs[-1]
            flow_output = final_output.get("candidate_residual_flow")
            if flow_output is None:
                raise RuntimeError(
                    "Missing candidate residual-flow output during training"
                )
            refined_metrics = self._official_match_quality(
                flow_output["trajectories"], input_dict
            )
            with torch.no_grad():
                base_metrics = self._official_match_quality(
                    flow_output["base_trajectories"], input_dict
                )
            formal_stage = (
                self.train_stage
                == "candidate_bank_residual_flow_full_convergence"
            )
            formal_phase = (
                self._jfer_full_convergence_phase(input_dict)
                if formal_stage
                else None
            )
            flow_loss, flow_metrics = self.candidate_residual_flow.get_loss(
                output=flow_output,
                input_dict=input_dict,
                refined_metrics=refined_metrics,
                base_metrics=base_metrics,
                loss_mode=(
                    "geometry"
                    if (
                        formal_stage and formal_phase == 0
                    ) or self.train_stage
                    in {
                        "candidate_bank_residual_flow_geometry",
                        "candidate_bank_residual_flow_generation_joint",
                    }
                    else "score"
                    if self.train_stage
                    == "candidate_bank_residual_flow_score"
                    else "joint"
                ),
            )

            base_weight = float(
                self.model_cfg.get("LOSS_WEIGHT_RESIDUAL_FLOW_BASE", 0.0)
            )
            if formal_stage and formal_phase < 2:
                base_weight = 0.0
            if base_weight > 0.0:
                base_outputs = self._base_generation_outputs(outputs)
                base_loss, tb = super().get_loss(
                    state, base_outputs, input_dict
                )
            else:
                base_loss = flow_loss.new_zeros(())
                tb = {}
            bank_loss = flow_loss.new_zeros(())
            bank_metrics = {}
            if (
                self.train_stage
                == "candidate_bank_residual_flow_generation_joint"
                or (formal_stage and formal_phase >= 2)
            ):
                bank_ret = state.get("candidate_bank")
                if bank_ret is None:
                    raise RuntimeError(
                        "Residual-flow generation joint training requires "
                        "candidate-bank state"
                    )
                bank_loss, bank_metrics = self.candidate_bank.get_loss(
                    bank_ret, input_dict
                )
            expansion_loss = flow_loss.new_zeros(())
            expansion_metrics = {}
            if formal_stage and formal_phase >= 1:
                expansion_loss, expansion_metrics = (
                    self._candidate_expansion_loss(
                        final_output,
                        input_dict,
                        score_only=(formal_phase == 1),
                    )
                )
            total = (
                base_weight * base_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_CANDIDATE_RESIDUAL_FLOW", 1.0
                    )
                )
                * flow_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_RESIDUAL_FLOW_BANK", 0.0
                    )
                )
                * bank_loss
                + (
                    float(
                        self.model_cfg.get(
                            "LOSS_WEIGHT_RESIDUAL_FLOW_EXPANSION", 1.0
                        )
                    )
                    * expansion_loss
                    if formal_stage and formal_phase >= 1
                    else expansion_loss.new_zeros(())
                )
            )
            tb["loss_residual_flow_base"] = base_loss.item()
            tb["loss_residual_flow_bank"] = bank_loss.item()
            tb["loss_residual_flow_expansion"] = expansion_loss.item()
            if formal_stage:
                tb["jfer_full_convergence_phase"] = float(formal_phase)
            tb["loss_integrated_joint_world"] = total.item()
            tb["joint_oracle_ade"] = refined_metrics["ade"].amin(
                dim=1
            ).mean().item()
            tb["joint_oracle_fde"] = refined_metrics["fde"].amin(
                dim=1
            ).mean().item()
            for key, value in flow_metrics.items():
                tb[key] = (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
            for key, value in bank_metrics.items():
                tb[key] = (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
            for key, value in expansion_metrics.items():
                tb[key] = (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
            return total, tb
        if self.train_stage in {
            "candidate_bank_structured_world_warmup",
            "candidate_bank_structured_world_joint",
            "candidate_bank_energy_warmup",
            "candidate_bank_energy_joint",
        }:
            if self.structured_behavior_world is None:
                raise RuntimeError(
                    "Structured world training requires "
                    "STRUCTURED_BEHAVIOR_WORLD.ENABLED=True"
                )
            generation_outputs = self._base_generation_outputs(outputs)
            base_weight = float(
                self.model_cfg.get("LOSS_WEIGHT_STRUCTURED_BASE", 1.0)
            )
            if base_weight > 0.0:
                base_loss, tb = super().get_loss(
                    state, generation_outputs, input_dict
                )
            else:
                # Posterior calibration deliberately preserves the verified
                # generator. Avoid constructing a no-op decoder loss graph.
                reference = generation_outputs[-1]["joint_logits"]
                base_loss = reference.new_zeros(())
                tb = {}
            # The unified posterior ranks exactly the frozen six-candidate
            # deployment support. MTR trajectory regression remains aligned to
            # the six decoder modes that produced the base loss above.
            if "world_credit_posterior" in outputs[-1]:
                final_generation = dict(outputs[-1])
                final_generation["joint_trajs"] = outputs[-1][
                    "world_credit_selected_trajs"
                ]
                final_generation["joint_logits"] = outputs[-1][
                    "world_credit_posterior"
                ]["joint_logits"]
            else:
                final_generation = generation_outputs[-1]
            final_metrics = self._official_match_quality(
                final_generation["joint_trajs"], input_dict
            )
            if "joint_oracle_ade" not in tb:
                tb["joint_oracle_ade"] = (
                    final_metrics["ade"].amin(dim=1).mean().item()
                )
                tb["joint_oracle_fde"] = (
                    final_metrics["fde"].amin(dim=1).mean().item()
                )
            base_quality = None
            base_trajectories = final_generation.get(
                "structured_behavior_base_trajs"
            )
            if base_trajectories is not None:
                base_quality = self._official_match_quality(
                    base_trajectories, input_dict
                )["quality"]
            behavior_loss_kwargs = {
                "state": state,
                "final_output": final_generation,
                "input_dict": input_dict,
                "quality": final_metrics["quality"],
                "base_quality": base_quality,
            }
            if (
                self.structured_behavior_world_architecture
                in {
                    "supervised_joint_hypothesis",
                    "coverage_ordered_joint_hypothesis",
                    "candidate_energy_transport",
                    "balanced_coverage_transport",
                }
            ):
                behavior_loss_kwargs["official_metrics"] = final_metrics
            if self.structured_behavior_world_architecture in {
                "candidate_energy_transport",
                "balanced_coverage_transport",
            }:
                behavior_loss_kwargs["group_ids"] = official_ap_group_ids(
                    input_dict,
                    final_generation["joint_logits"].shape[0],
                    final_generation["joint_logits"].device,
                )
            behavior_loss, behavior_metrics = (
                self.structured_behavior_world.get_loss(
                    **behavior_loss_kwargs
                )
            )
            bank_loss = base_loss.new_zeros(())
            bank_metrics = {}
            bank_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_STRUCTURED_BANK", 0.0
                )
            )
            if bank_weight > 0.0:
                bank_loss, bank_metrics = self.candidate_bank.get_loss(
                    state["candidate_bank"], input_dict
                )
            total = (
                base_weight * base_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_STRUCTURED_BEHAVIOR", 1.0
                    )
                )
                * behavior_loss
                + bank_weight * bank_loss
            )
            tb["loss_structured_world_base"] = base_loss.item()
            tb["loss_structured_world_behavior"] = behavior_loss.item()
            tb["loss_structured_world_bank"] = bank_loss.item()
            tb["loss_integrated_joint_world"] = total.item()
            for metrics in (behavior_metrics, bank_metrics):
                for key, value in metrics.items():
                    tb[key] = (
                        value.detach().float().mean().item()
                        if torch.is_tensor(value)
                        else float(value)
                    )
            return total, tb
        if self.train_stage == "candidate_bank_reciprocal_generator":
            loss, metrics = self._candidate_expansion_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_reciprocal_selector":
            loss, metrics = self._interaction_set_replacement_loss(
                outputs[-1], input_dict, reciprocal=True
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_reciprocal_world_set_decoder":
            loss, metrics = self._reciprocal_world_set_decoder_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_reciprocal_world_set_joint":
            generator_loss, generator_metrics = self._candidate_expansion_loss(
                outputs[-1], input_dict
            )
            decoder_loss, decoder_metrics = (
                self._reciprocal_world_set_decoder_loss(
                    outputs[-1], input_dict
                )
            )
            total = (
                float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_RECIPROCAL_GENERATOR", 0.5
                    )
                )
                * generator_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_RECIPROCAL_WORLD_SET", 1.0
                    )
                )
                * decoder_loss
            )
            metrics = dict(generator_metrics)
            metrics.update(decoder_metrics)
            metrics["loss_reciprocal_generator_total"] = generator_loss
            metrics["loss_reciprocal_world_set_total"] = decoder_loss
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = total.item()
            return total, tb
        if self.train_stage == "candidate_bank_reciprocal_relative":
            generator_loss, generator_metrics = (
                self._candidate_expansion_loss(outputs[-1], input_dict)
            )
            selector_loss, selector_metrics = (
                self._interaction_set_replacement_loss(
                    outputs[-1], input_dict, reciprocal=True
                )
            )
            total = (
                float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_RECIPROCAL_GENERATOR", 1.0
                    )
                )
                * generator_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_RECIPROCAL_RELATIVE_SELECTOR", 1.0
                    )
                )
                * selector_loss
            )
            metrics = dict(generator_metrics)
            metrics.update(selector_metrics)
            metrics["loss_reciprocal_generator_total"] = generator_loss
            metrics["loss_reciprocal_selector_total"] = selector_loss
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = total.item()
            return total, tb
        if self.train_stage == "candidate_bank_interaction_set_replacement":
            loss, metrics = self._interaction_set_replacement_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_interaction_time_warp":
            loss, metrics = self._interaction_time_warp_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_deployed_confidence_joint":
            loss, metrics = self._end_to_end_deployed_set_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["joint_oracle_ade"] = tb[
                "end_to_end_deployed_oracle_ade"
            ]
            tb["joint_oracle_fde"] = tb[
                "end_to_end_deployed_oracle_fde"
            ]
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb

        if self.train_stage == "candidate_bank_set_to_slot_retrieval":
            transport_loss, transport_metrics = (
                self._set_to_slot_transport_loss(outputs[-1], input_dict)
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in transport_metrics.items()
            }
            tb["loss_integrated_joint_world"] = transport_loss.item()
            return transport_loss, tb

        if (
            self.train_stage
            == "candidate_bank_deployed_geometry_refinement"
        ):
            loss, metrics = self._deployed_trajectory_refiner_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_final_permutation_calibration":
            loss, metrics = self._deployed_set_permutation_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage == "candidate_bank_set_utility_calibration":
            loss, metrics = self._deployed_set_utility_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage in {
            "candidate_bank_ap_calibration",
            "candidate_bank_permutation_calibration",
        }:
            metric_output = dict(outputs[-1])
            metric_modes = metric_output[
                "metric_aligned_scorer"
            ]["horizon_logits"].shape[1]
            metric_output["joint_trajs"] = metric_output[
                "joint_trajs"
            ][:, :metric_modes]
            metric_output["joint_logits"] = metric_output[
                "joint_logits"
            ][:, :metric_modes]
            loss, metrics = self._metric_aligned_scorer_loss(
                metric_output, input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb
        if self.train_stage in {
            "candidate_bank_expansion",
            "candidate_bank_expansion_calibration",
        }:
            loss, metrics = self._candidate_expansion_loss(
                outputs[-1], input_dict
            )
            tb = {
                key: (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
                for key, value in metrics.items()
            }
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb

        if self.train_stage == "candidate_bank_slot_transport_joint":
            # Keep the verified expansion/deployment branch in the forward
            # graph, but optimize only the six trajectories that feed it and
            # the dense candidate-bank transport objective.  The expansion
            # selector and its calibrated logits remain frozen.
            protected_modes = self.num_output_modes
            base_outputs = list(outputs)
            base_final = dict(outputs[-1])
            base_final["joint_trajs"] = base_final["joint_trajs"][
                :, :protected_modes
            ]
            protected_logits = base_final.get(
                "protected_base_joint_logits",
                base_final["joint_logits"],
            )
            base_final["joint_logits"] = protected_logits[
                :, :protected_modes
            ]
            base_outputs[-1] = base_final
            base_loss, tb = super().get_loss(
                state, base_outputs, input_dict
            )
            bank_ret = state.get("candidate_bank")
            if bank_ret is None:
                raise RuntimeError(
                    "candidate_bank_slot_transport_joint requires "
                    "candidate-bank state"
                )
            bank_loss, bank_metrics = self.candidate_bank.get_loss(
                bank_ret, input_dict
            )
            bank_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_SLOT_TRANSPORT_BANK", 1.0
                )
            )
            total = base_loss + bank_weight * bank_loss
            tb["loss_slot_transport_base"] = base_loss.item()
            tb["loss_slot_transport_bank"] = bank_loss.item()
            tb["loss_integrated_joint_world"] = total.item()
            for key, value in bank_metrics.items():
                tb[key] = (
                    value.detach().float().mean().item()
                    if torch.is_tensor(value)
                    else float(value)
                )
            return total, tb

        if self.train_stage == "candidate_bank_set_to_slot_joint":
            protected_modes = self.num_output_modes
            base_outputs = list(outputs)
            base_final = dict(outputs[-1])
            base_final["joint_trajs"] = base_final["joint_trajs"][
                :, :protected_modes
            ]
            base_final["joint_logits"] = base_final["joint_logits"][
                :, :protected_modes
            ]
            base_outputs[-1] = base_final
            base_loss, tb = super().get_loss(
                state, base_outputs, input_dict
            )
            bank_ret = state.get("candidate_bank")
            if bank_ret is None:
                raise RuntimeError(
                    "candidate_bank_set_to_slot_joint requires bank state"
                )
            bank_loss, bank_metrics = self.candidate_bank.get_loss(
                bank_ret, input_dict
            )
            expansion_loss, expansion_metrics = (
                self._candidate_expansion_loss(outputs[-1], input_dict)
            )
            transport_loss, transport_metrics = (
                self._set_to_slot_transport_loss(
                    outputs[-1], input_dict
                )
            )

            metric_loss = base_loss.new_zeros(())
            metric_metrics = {}
            if self.metric_scorer is not None:
                metric_output = dict(base_final)
                metric_modes = metric_output[
                    "metric_aligned_scorer"
                ]["horizon_logits"].shape[1]
                metric_output["joint_trajs"] = metric_output[
                    "joint_trajs"
                ][:, :metric_modes]
                metric_loss, metric_metrics = (
                    self._metric_aligned_scorer_loss(
                        metric_output, input_dict
                    )
                )

            total = (
                float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_SET_TO_SLOT_BASE", 1.0
                    )
                )
                * base_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_SET_TO_SLOT_BANK", 0.5
                    )
                )
                * bank_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_SET_TO_SLOT_EXPANSION", 0.5
                    )
                )
                * expansion_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_SET_TO_SLOT_TRANSPORT", 1.0
                    )
                )
                * transport_loss
                + float(
                    self.model_cfg.get(
                        "LOSS_WEIGHT_SET_TO_SLOT_METRIC", 0.10
                    )
                )
                * metric_loss
            )
            tb["loss_set_to_slot_base"] = base_loss.item()
            tb["loss_set_to_slot_bank"] = bank_loss.item()
            tb["loss_set_to_slot_expansion"] = expansion_loss.item()
            tb["loss_set_to_slot_metric"] = metric_loss.item()
            tb["loss_integrated_joint_world"] = total.item()
            for metrics_dict in (
                bank_metrics,
                expansion_metrics,
                transport_metrics,
                metric_metrics,
            ):
                for key, value in metrics_dict.items():
                    tb[key] = (
                        value.detach().float().mean().item()
                        if torch.is_tensor(value)
                        else float(value)
                    )
            return total, tb

        if self.train_stage == "candidate_bank_end_to_end":
            # Close the training/deployment loop.  Earlier candidate-bank
            # stages optimized geometry, admission, and confidence in
            # isolation; the deployed 12-to-6 decision therefore never saw a
            # single joint objective.  This stage keeps the trusted base loss
            # while jointly supervising every branch used at inference.
            protected_modes = self.num_output_modes
            base_outputs = list(outputs)
            base_final = dict(outputs[-1])
            base_final["joint_trajs"] = base_final["joint_trajs"][
                :, :protected_modes
            ]
            base_final["joint_logits"] = base_final["joint_logits"][
                :, :protected_modes
            ]
            base_outputs[-1] = base_final
            base_loss, tb = super().get_loss(
                state, base_outputs, input_dict
            )
            bank_ret = state.get("candidate_bank")
            if bank_ret is None:
                raise RuntimeError(
                    "candidate_bank_end_to_end requires candidate-bank state"
                )
            bank_loss, bank_metrics = self.candidate_bank.get_loss(
                bank_ret, input_dict
            )
            expansion_loss, expansion_metrics = (
                self._candidate_expansion_loss(outputs[-1], input_dict)
            )

            metric_loss = base_loss.new_zeros(())
            metric_metrics = {}
            if self.metric_scorer is not None:
                # The metric scorer is attached before six expansion modes
                # are appended.  Slice only the matching base trajectories so
                # its 3/5/8-second targets stay shape-aligned.
                metric_output = dict(base_final)
                metric_modes = metric_output[
                    "metric_aligned_scorer"
                ]["horizon_logits"].shape[1]
                metric_output["joint_trajs"] = metric_output[
                    "joint_trajs"
                ][:, :metric_modes]
                metric_loss, metric_metrics = (
                    self._metric_aligned_scorer_loss(
                        metric_output, input_dict
                    )
                )

            deployed_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_END_TO_END_DEPLOYED_SET", 0.0
                )
            )
            deployed_loss = base_loss.new_zeros(())
            deployed_metrics = {}
            if deployed_weight > 0.0:
                deployed_loss, deployed_metrics = (
                    self._end_to_end_deployed_set_loss(
                        outputs[-1], input_dict
                    )
                )

            base_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_END_TO_END_BASE", 1.0
                )
            )
            bank_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_END_TO_END_BANK", 1.0
                )
            )
            expansion_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_END_TO_END_EXPANSION", 0.5
                )
            )
            metric_weight = float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_END_TO_END_METRIC", 0.25
                )
            )
            total = (
                base_weight * base_loss
                + bank_weight * bank_loss
                + expansion_weight * expansion_loss
                + metric_weight * metric_loss
                + deployed_weight * deployed_loss
            )
            tb["loss_end_to_end_base"] = base_loss.item()
            tb["loss_end_to_end_bank"] = bank_loss.item()
            tb["loss_end_to_end_expansion"] = expansion_loss.item()
            tb["loss_end_to_end_metric"] = metric_loss.item()
            tb["loss_end_to_end_deployed"] = deployed_loss.item()
            tb["loss_integrated_joint_world"] = total.item()
            for metrics in (
                bank_metrics,
                expansion_metrics,
                metric_metrics,
                deployed_metrics,
            ):
                for key, value in metrics.items():
                    tb[key] = (
                        value.detach().float().mean().item()
                        if torch.is_tensor(value)
                        else float(value)
                    )
            return total, tb

        base_loss, tb = super().get_loss(state, outputs, input_dict)
        bank_ret = state.get("candidate_bank")
        if bank_ret is None:
            return base_loss, tb
        bank_loss, bank_metrics = self.candidate_bank.get_loss(
            bank_ret, input_dict
        )
        total = base_loss + bank_loss
        tb["loss_integrated_joint_world"] = total.item()
        for key, value in bank_metrics.items():
            tb[key] = (
                value.item() if torch.is_tensor(value) else float(value)
            )
        return total, tb
