"""Core components of Joint Future Exploration and Reasoning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_reasoning import _build_mlp
from .spatial_context import SPATIAL_EVIDENCE_DIM

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
