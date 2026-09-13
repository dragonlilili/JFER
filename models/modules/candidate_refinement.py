"""Proposal-anchored world-conditioned residual flow.

The module keeps the trusted MTR candidate set as an explicit geometric prior.
It learns a bounded correction field and candidate success energy jointly, so
world reasoning can improve trajectories without replacing the proposal model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_context import SPATIAL_EVIDENCE_DIM
from .horizon_scoring import build_soft_map_credit_targets


def _mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class WorldConditionedCandidateResidualFlow(nn.Module):
    """Refine and rank a proposal bank with a bounded conditional flow.

    The output is an exact identity at initialization because the geometric and
    confidence residual heads are zero initialized. After that, deployment is
    candidate-specific: a supervised utility gate decides which proposals may
    move instead of forcing every scene through one global scalar gate.
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
        self.model_cfg = cfg
        self.hidden_dim = int(hidden_dim)
        self.model_dim = int(cfg.get("MODEL_DIM", 128))
        self.num_future_frames = int(num_future_frames)
        self.dt = float(dt)
        self.num_flow_steps = int(cfg.get("NUM_FLOW_STEPS", 2))
        self.max_score_delta = float(cfg.get("MAX_SCORE_DELTA", 1.5))
        self.eps = 1e-6

        steps = torch.as_tensor(measurement_steps, dtype=torch.long)
        if steps.numel() != 3 or int(steps[-1]) >= self.num_future_frames:
            raise ValueError(
                "Candidate residual flow expects valid 3/5/8-second steps"
            )
        self.register_buffer("measurement_steps", steps, persistent=False)

        num_temporal_tokens = int(cfg.get("NUM_TEMPORAL_TOKENS", 16))
        if num_temporal_tokens < 4:
            raise ValueError("NUM_TEMPORAL_TOKENS must be at least four")
        temporal_indices = torch.linspace(
            0,
            self.num_future_frames - 1,
            num_temporal_tokens,
        ).round().long().unique(sorted=True)
        self.num_temporal_tokens = int(temporal_indices.numel())
        self.register_buffer(
            "temporal_indices", temporal_indices, persistent=False
        )

        max_deltas = torch.as_tensor(
            cfg.get("MAX_HORIZON_DELTAS", [0.35, 0.75, 1.50]),
            dtype=torch.float32,
        )
        if max_deltas.numel() != 3 or bool((max_deltas <= 0).any()):
            raise ValueError("MAX_HORIZON_DELTAS must contain 3 positives")
        self.register_buffer(
            "max_horizon_deltas", max_deltas, persistent=False
        )
        self.register_buffer(
            "dense_max_delta",
            self._interpolate_horizon_values(max_deltas),
            persistent=False,
        )
        token_max = self.dense_max_delta.index_select(0, temporal_indices)
        self.register_buffer("token_max_delta", token_max, persistent=False)

        horizon_weights = torch.as_tensor(
            cfg.get("HORIZON_WEIGHTS", [0.20, 0.30, 0.50]),
            dtype=torch.float32,
        )
        if horizon_weights.numel() != 3 or float(horizon_weights.sum()) <= 0:
            raise ValueError("HORIZON_WEIGHTS must contain 3 valid values")
        self.register_buffer(
            "horizon_weights",
            horizon_weights / horizon_weights.sum(),
            persistent=False,
        )

        dropout = float(cfg.get("DROPOUT", 0.05))
        heads = int(cfg.get("NUM_HEADS", min(num_heads, 4)))
        if self.model_dim % heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")

        # Two-agent position/velocity and relative interaction kinematics.
        self.trajectory_proj = _mlp(14, self.model_dim, self.model_dim, dropout)
        self.world_proj = nn.Linear(self.hidden_dim, self.model_dim)
        self.scene_proj = nn.Linear(self.hidden_dim, self.model_dim)
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(SPATIAL_EVIDENCE_DIM),
            nn.Linear(SPATIAL_EVIDENCE_DIM, self.model_dim),
        )
        self.score_proj = _mlp(3, self.model_dim, self.model_dim, dropout)
        self.type_embedding = nn.Embedding(3, self.model_dim)
        self.branch_embedding = nn.Embedding(2, self.model_dim)
        self.time_embedding = nn.Parameter(
            torch.empty(self.num_temporal_tokens, self.model_dim)
        )
        self.flow_time_proj = _mlp(1, self.model_dim, self.model_dim, dropout)
        nn.init.normal_(self.time_embedding, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=heads,
            dim_feedforward=(
                self.model_dim * int(cfg.get("TEMPORAL_FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(cfg.get("NUM_TEMPORAL_LAYERS", 2)),
            norm=nn.LayerNorm(self.model_dim),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=heads,
            dim_feedforward=(
                self.model_dim * int(cfg.get("SET_FFN_MULTIPLIER", 2))
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=int(cfg.get("NUM_SET_LAYERS", 2)),
            norm=nn.LayerNorm(self.model_dim),
        )
        self.temporal_set_fusion = _mlp(
            2 * self.model_dim,
            2 * self.model_dim,
            self.model_dim,
            dropout,
        )

        self.flow_head = _mlp(
            self.model_dim, self.model_dim, 4, dropout
        )
        self.candidate_gate_head = _mlp(
            self.model_dim, self.model_dim, 1, dropout
        )
        self.horizon_score_head = _mlp(
            self.model_dim, self.model_dim, 1, dropout
        )

        improvement_cfg = cfg.get("IMPROVEMENT_GATED", {})
        self.improvement_cfg = improvement_cfg
        self.use_improvement_gated = bool(
            improvement_cfg.get("ENABLED", False)
        )
        if self.use_improvement_gated:
            self.improvement_alpha_head = _mlp(
                self.model_dim,
                self.model_dim,
                1,
                float(improvement_cfg.get("DROPOUT", dropout)),
            )
            self.improvement_gain_head = _mlp(
                self.model_dim,
                self.model_dim,
                1,
                float(improvement_cfg.get("DROPOUT", dropout)),
            )
            self.register_buffer(
                "improvement_alpha_basis",
                self._build_horizon_basis(steps, self.num_future_frames),
                persistent=False,
            )
            nn.init.zeros_(self.improvement_alpha_head[-1].weight)
            nn.init.constant_(
                self.improvement_alpha_head[-1].bias,
                float(improvement_cfg.get("ALPHA_BIAS_INIT", -2.0)),
            )
            nn.init.zeros_(self.improvement_gain_head[-1].weight)
            nn.init.zeros_(self.improvement_gain_head[-1].bias)
        else:
            self.improvement_alpha_head = None
            self.improvement_gain_head = None

        # Zero residual heads make the complete model an exact identity at load
        # time while retaining direct, non-zero loss gradients on both heads.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
        nn.init.zeros_(self.candidate_gate_head[-1].weight)
        nn.init.zeros_(self.candidate_gate_head[-1].bias)
        nn.init.zeros_(self.horizon_score_head[-1].weight)
        nn.init.zeros_(self.horizon_score_head[-1].bias)
        nn.init.zeros_(self.type_embedding.weight)
        nn.init.zeros_(self.branch_embedding.weight)

    @staticmethod
    def _build_horizon_basis(steps, num_future_frames):
        """Interpolate three horizon gates from zero at the current frame."""
        basis = torch.zeros(
            num_future_frames, steps.numel(), dtype=torch.float32
        )
        anchors = [int(value) for value in steps.tolist()]
        for frame in range(num_future_frames):
            if frame <= anchors[0]:
                basis[frame, 0] = (frame + 1) / (anchors[0] + 1)
                continue
            assigned = False
            for horizon in range(1, len(anchors)):
                if frame <= anchors[horizon]:
                    width = max(anchors[horizon] - anchors[horizon - 1], 1)
                    alpha = (frame - anchors[horizon - 1]) / width
                    basis[frame, horizon - 1] = 1.0 - alpha
                    basis[frame, horizon] = alpha
                    assigned = True
                    break
            if not assigned:
                basis[frame, -1] = 1.0
        return basis

    def improvement_phase(self, epoch):
        """Return proposal, gate, or joint phase for the four-epoch protocol."""
        proposal_epochs = int(
            self.improvement_cfg.get("STAGE_A_EPOCHS", 2)
        )
        gate_epochs = int(self.improvement_cfg.get("STAGE_B_EPOCHS", 1))
        epoch = int(epoch)
        if epoch < proposal_epochs:
            return 0
        if epoch < proposal_epochs + gate_epochs:
            return 1
        return 2

    def _improvement_proposal_modules(self):
        return (
            self.trajectory_proj,
            self.world_proj,
            self.scene_proj,
            self.spatial_proj,
            self.score_proj,
            self.type_embedding,
            self.branch_embedding,
            self.flow_time_proj,
            self.temporal_encoder,
            self.candidate_set_encoder,
            self.temporal_set_fusion,
            self.flow_head,
        )

    def set_improvement_train_phase(self, phase):
        """Apply the Stage A/B/C parameter and module-mode isolation."""
        if not self.use_improvement_gated:
            raise RuntimeError("Improvement-gated JFER is not enabled")
        train_proposal = int(phase) in (0, 2)
        train_gate = int(phase) in (1, 2)
        for module in self._improvement_proposal_modules():
            module.train(train_proposal)
            module.requires_grad_(train_proposal)
        for module in (
            self.improvement_alpha_head,
            self.improvement_gain_head,
        ):
            module.train(train_gate)
            module.requires_grad_(train_gate)
        # These legacy JFER heads are outside the fixed-score v2 deployment.
        for module in (self.candidate_gate_head, self.horizon_score_head):
            module.eval()
            module.requires_grad_(False)

    def forward_improvement_gated(
        self,
        world_hidden,
        trajectories,
        base_logits,
        selection_logits,
        branch_ids,
        pair_type_ids,
        scene_context,
        spatial_evidence,
        pair_state,
        phase,
    ):
        """Refine an already selected six-mode set without touching scores."""
        if not self.use_improvement_gated:
            raise RuntimeError("Improvement-gated JFER is not enabled")
        flow_time = trajectories.new_full(
            trajectories.shape[:2],
            float(self.improvement_cfg.get("PROPOSAL_FLOW_TIME", 0.5)),
        )
        temporal, candidate = self._encode(
            trajectories=trajectories,
            world_hidden=world_hidden,
            base_logits=base_logits,
            selection_logits=selection_logits,
            branch_ids=branch_ids,
            pair_type_ids=pair_type_ids,
            scene_context=scene_context,
            spatial_evidence=spatial_evidence,
            pair_state=pair_state,
            flow_time=flow_time,
        )
        token_delta = self._predict_flow(temporal)
        proposal_delta = self._interpolate_dense(token_delta)
        proposal_trajectories = trajectories + proposal_delta

        horizon_indices = torch.searchsorted(
            self.temporal_indices, self.measurement_steps
        ).clamp_max(self.num_temporal_tokens - 1)
        horizon_hidden = temporal.index_select(2, horizon_indices)
        alpha_logits = self.improvement_alpha_head(horizon_hidden).squeeze(-1)
        gain_prediction = self.improvement_gain_head(
            horizon_hidden
        ).squeeze(-1)
        if int(phase) == 0:
            alpha_horizon = trajectories.new_full(
                alpha_logits.shape,
                float(self.improvement_cfg.get("STAGE_A_ALPHA", 1.0)),
            )
        else:
            alpha_horizon = torch.sigmoid(alpha_logits)
        alpha_dense = torch.einsum(
            "th,bmh->bmt",
            self.improvement_alpha_basis.type_as(alpha_horizon),
            alpha_horizon,
        ).clamp(0.0, 1.0)
        effect_enabled = bool(
            self.improvement_cfg.get("EFFECT_ENABLED", True)
        )
        effect_scale = 1.0 if effect_enabled else 0.0
        deployed_delta = (
            effect_scale
            * alpha_dense[:, :, None, :, None]
            * proposal_delta
        )
        deployed_trajectories = trajectories + deployed_delta
        return {
            "trajectories": deployed_trajectories,
            "base_trajectories": trajectories,
            "proposal_trajectories": proposal_trajectories,
            "proposal_delta": proposal_delta,
            "deployed_delta": deployed_delta,
            "token_delta": token_delta,
            "alpha_logits": alpha_logits,
            "alpha_horizon": alpha_horizon,
            "alpha_dense": alpha_dense,
            "gain_prediction": gain_prediction,
            "horizon_hidden": horizon_hidden,
            "candidate_hidden": candidate,
            "phase": int(phase),
            "effect_enabled": effect_enabled,
        }

    def _interpolate_horizon_values(self, values):
        anchors = [0] + [int(value) + 1 for value in self.measurement_steps]
        start = values.new_tensor(float(self.model_cfg.get("START_DELTA", 0.05)))
        anchor_values = torch.cat([start[None], values])
        dense = values.new_zeros(self.num_future_frames)
        for frame in range(1, self.num_future_frames + 1):
            for segment in range(1, len(anchors)):
                if frame <= anchors[segment]:
                    alpha = (frame - anchors[segment - 1]) / max(
                        anchors[segment] - anchors[segment - 1], 1
                    )
                    dense[frame - 1] = (
                        (1.0 - alpha) * anchor_values[segment - 1]
                        + alpha * anchor_values[segment]
                    )
                    break
            else:
                dense[frame - 1] = values[-1]
        return dense

    @staticmethod
    def _normalize_score(score):
        centered = score - score.mean(dim=-1, keepdim=True)
        scale = centered.detach().std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        return centered / scale

    def _trajectory_features(self, trajectories, pair_state):
        velocity = torch.zeros_like(trajectories)
        velocity[..., 1:, :] = (
            trajectories[..., 1:, :] - trajectories[..., :-1, :]
        ) / self.dt
        velocity[..., 0, :] = pair_state[:, None, :, 2:4].type_as(
            trajectories
        )
        position = trajectories.index_select(3, self.temporal_indices)
        velocity = velocity.index_select(3, self.temporal_indices)
        position = position.permute(0, 1, 3, 2, 4)
        velocity = velocity.permute(0, 1, 3, 2, 4)
        relative_position = position[..., 1, :] - position[..., 0, :]
        relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
        distance = torch.linalg.vector_norm(
            relative_position, dim=-1, keepdim=True
        )
        closing = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(0.5)
        return torch.cat(
            [
                position.flatten(start_dim=-2) / 50.0,
                velocity.flatten(start_dim=-2) / 20.0,
                relative_position / 20.0,
                relative_velocity / 20.0,
                distance / 20.0,
                closing / 10.0,
            ],
            dim=-1,
        )

    def _horizon_to_temporal(self, tensor):
        if tensor.shape[2] != 3:
            raise ValueError("Expected three horizon features")
        horizon_index = torch.bucketize(
            self.temporal_indices,
            self.measurement_steps,
            right=True,
        ).clamp_max(2)
        return tensor.index_select(2, horizon_index)

    def _encode(
        self,
        trajectories,
        world_hidden,
        base_logits,
        selection_logits,
        branch_ids,
        pair_type_ids,
        scene_context,
        spatial_evidence,
        pair_state,
        flow_time,
    ):
        batch_size, num_modes = base_logits.shape
        if world_hidden.shape[:3] != (batch_size, num_modes, 3):
            raise ValueError("world_hidden must have shape [B, M, 3, H]")
        score_feature = torch.stack(
            [
                self._normalize_score(base_logits),
                self._normalize_score(selection_logits),
                branch_ids.type_as(base_logits),
            ],
            dim=-1,
        )
        pair_type_ids = pair_type_ids.long().clamp(0, 2)
        candidate_context = (
            self.world_proj(world_hidden).mean(dim=2)
            + self.scene_proj(scene_context)[:, None]
            + self.score_proj(score_feature)
            + self.type_embedding(pair_type_ids)[:, None]
            + self.branch_embedding(branch_ids.long().clamp(0, 1))
        )
        temporal = (
            self.trajectory_proj(
                self._trajectory_features(trajectories, pair_state)
            )
            + self._horizon_to_temporal(self.world_proj(world_hidden))
            + self._horizon_to_temporal(
                self.spatial_proj(spatial_evidence.type_as(world_hidden))
            )
            + candidate_context[:, :, None]
            + self.time_embedding[None, None]
        )
        if flow_time.ndim == 1:
            flow_time = flow_time[:, None].expand(-1, num_modes)
        temporal = temporal + self.flow_time_proj(flow_time[..., None])[
            :, :, None
        ]
        temporal = self.temporal_encoder(
            temporal.reshape(
                batch_size * num_modes,
                self.num_temporal_tokens,
                self.model_dim,
            )
        ).reshape(
            batch_size,
            num_modes,
            self.num_temporal_tokens,
            self.model_dim,
        )
        candidate = self.candidate_set_encoder(
            temporal.mean(dim=2) + candidate_context
        )
        temporal = temporal + self.temporal_set_fusion(
            torch.cat(
                [temporal, candidate[:, :, None].expand_as(temporal)], dim=-1
            )
        )
        return temporal, candidate

    def _predict_flow(self, temporal):
        raw = torch.tanh(self.flow_head(temporal)).reshape(
            temporal.shape[0],
            temporal.shape[1],
            self.num_temporal_tokens,
            2,
            2,
        )
        return raw * self.token_max_delta.type_as(raw)[None, None, :, None, None]

    def _interpolate_dense(self, token_delta):
        batch_size, num_modes, _, num_agents, num_xy = token_delta.shape
        flat = token_delta.permute(0, 1, 3, 4, 2).reshape(
            batch_size * num_modes, num_agents * num_xy, -1
        )
        dense = F.interpolate(
            flat,
            size=self.num_future_frames,
            mode="linear",
            align_corners=True,
        )
        return dense.reshape(
            batch_size, num_modes, num_agents, num_xy, self.num_future_frames
        ).permute(0, 1, 2, 4, 3).contiguous()

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
        pair_state,
        enable_scoring=True,
    ):
        base_trajectories = trajectories
        current = trajectories
        last_temporal = None
        last_candidate = None
        last_token_flow = None
        candidate_gate = None
        candidate_gate_logits = None
        for step in range(self.num_flow_steps):
            flow_time = trajectories.new_full(
                (trajectories.shape[0], trajectories.shape[1]),
                (step + 0.5) / self.num_flow_steps,
            )
            last_temporal, last_candidate = self._encode(
                trajectories=current,
                world_hidden=world_hidden,
                base_logits=base_logits,
                selection_logits=selection_logits,
                branch_ids=branch_ids,
                pair_type_ids=pair_type_ids,
                scene_context=scene_context,
                spatial_evidence=spatial_evidence,
                pair_state=pair_state,
                flow_time=flow_time,
            )
            last_token_flow = self._predict_flow(last_temporal)
            candidate_gate_logits = self.candidate_gate_head(
                last_candidate
            ).squeeze(-1)
            candidate_gate = torch.sigmoid(candidate_gate_logits)
            dense_flow = self._interpolate_dense(last_token_flow)
            current = current + (
                candidate_gate[:, :, None, None, None]
                * dense_flow
                / max(self.num_flow_steps, 1)
            )

        horizon_indices = torch.searchsorted(
            self.temporal_indices,
            self.measurement_steps,
        ).clamp_max(self.num_temporal_tokens - 1)
        horizon_hidden = last_temporal.index_select(2, horizon_indices)
        if enable_scoring:
            raw_horizon_logits = self.horizon_score_head(
                horizon_hidden
            ).squeeze(-1)
            aggregate = (
                raw_horizon_logits
                * self.horizon_weights.type_as(raw_horizon_logits)[None, None]
            ).sum(dim=-1)
            aggregate = aggregate - aggregate.mean(dim=-1, keepdim=True)
            score_delta = self.max_score_delta * torch.tanh(aggregate)
        else:
            raw_horizon_logits = base_logits.new_zeros(
                base_logits.shape[0], base_logits.shape[1], 3
            )
            score_delta = torch.zeros_like(base_logits)

        refined_logits = base_logits + score_delta
        refined_selection_logits = selection_logits + score_delta
        return {
            "base_trajectories": base_trajectories,
            "trajectories": current,
            "base_logits": base_logits,
            "joint_logits": refined_logits,
            "selection_logits": refined_selection_logits,
            "raw_horizon_logits": raw_horizon_logits,
            "score_delta": score_delta,
            "admission_delta": score_delta,
            "token_flow": last_token_flow,
            "dense_delta": current - base_trajectories,
            "candidate_gate": candidate_gate,
            "candidate_gate_logits": candidate_gate_logits,
            "flow_gain": trajectories.new_ones(()),
            "score_gain": trajectories.new_ones(()),
            "candidate_hidden": last_candidate,
            "horizon_hidden": horizon_hidden,
            "context": {
                "world_hidden": world_hidden,
                "base_logits": base_logits,
                "selection_logits": selection_logits,
                "branch_ids": branch_ids,
                "pair_type_ids": pair_type_ids,
                "scene_context": scene_context,
                "spatial_evidence": spatial_evidence,
                "pair_state": pair_state,
            },
        }

    @staticmethod
    def _weighted_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def get_loss(
        self,
        output,
        input_dict,
        refined_metrics,
        base_metrics,
        loss_mode="joint",
    ):
        refined = output["trajectories"]
        base = output["base_trajectories"].detach()
        dtype = refined.dtype
        pair_valid = refined_metrics["pair_valid"].type_as(refined)
        sample_valid = pair_valid.any(dim=-1).type_as(refined)

        # Up to three distinct candidates receive geometric responsibility,
        # one for each official horizon. Other modes keep their identity.
        base_cost = base_metrics["horizon_cost"].detach()
        winner = base_cost.argmin(dim=1)
        responsibility = base_cost.new_zeros(base_cost.shape[:2])
        horizon_weight = self.horizon_weights.type_as(base_cost)
        for horizon in range(3):
            responsibility.scatter_add_(
                1,
                winner[:, horizon : horizon + 1],
                (
                    pair_valid[:, horizon : horizon + 1]
                    * horizon_weight[horizon]
                ),
            )

        gt = input_dict["pair_gt_trajs"].to(refined.device).type_as(refined)[
            ..., 0:2
        ]
        gt_mask = input_dict["pair_gt_trajs_mask"].to(refined.device).bool()
        target = gt[:, None].expand_as(base)
        tau = torch.rand(
            base.shape[0], base.shape[1], device=base.device, dtype=dtype
        )
        active = (responsibility > 0).type_as(base)
        interpolated = base + (
            tau[:, :, None, None, None]
            * active[:, :, None, None, None]
            * (target - base)
        )
        context = output["context"]
        temporal, _ = self._encode(
            trajectories=interpolated,
            flow_time=tau,
            **context,
        )
        predicted_flow = self._interpolate_dense(self._predict_flow(temporal))
        max_delta = self.dense_max_delta.type_as(base)[None, None, None, :, None]
        target_flow = max_delta * torch.tanh((target - base) / max_delta)
        flow_error = F.smooth_l1_loss(
            predicted_flow,
            target_flow,
            reduction="none",
            beta=float(self.model_cfg.get("FLOW_BETA", 0.5)),
        ).sum(dim=-1)
        flow_valid = gt_mask[:, None].type_as(flow_error)
        flow_per_mode = (flow_error * flow_valid).sum(dim=(-1, -2)) / (
            flow_valid.sum(dim=(-1, -2)).clamp_min(1.0)
        )
        loss_flow = self._weighted_mean(
            flow_per_mode,
            responsibility * sample_valid[:, None],
        )

        gate_target = (responsibility > 0).type_as(refined)
        gate_pos_weight = float(
            self.model_cfg.get("GATE_POSITIVE_WEIGHT", 3.0)
        )
        gate_bce = F.binary_cross_entropy_with_logits(
            output["candidate_gate_logits"], gate_target, reduction="none"
        )
        gate_bce = gate_bce * (
            1.0 + (gate_pos_weight - 1.0) * gate_target
        )
        loss_gate = self._weighted_mean(
            gate_bce.mean(dim=-1), sample_valid
        )

        base_best = base_metrics["ade"].detach().argmin(dim=-1)
        batch_idx = torch.arange(base.shape[0], device=base.device)
        winner_ade = refined_metrics["ade"][batch_idx, base_best]
        loss_winner_ade = self._weighted_mean(winner_ade, sample_valid)

        coverage_temperature = max(
            float(self.model_cfg.get("COVERAGE_TEMPERATURE", 0.20)), self.eps
        )
        refined_cost = refined_metrics["horizon_cost"]
        soft_best_cost = -coverage_temperature * torch.logsumexp(
            -refined_cost / coverage_temperature, dim=1
        )
        loss_coverage = self._weighted_mean(
            (
                soft_best_cost
                * pair_valid
                * self.horizon_weights.type_as(refined)[None]
            ).sum(dim=-1),
            sample_valid,
        )

        base_best_cost = base_cost.amin(dim=1)
        refined_best_cost = refined_cost.amin(dim=1)
        no_harm = F.relu(
            refined_best_cost
            - base_best_cost
            - float(self.model_cfg.get("NO_HARM_TOLERANCE", 0.02))
        )
        loss_no_harm = self._weighted_mean(
            (no_harm * pair_valid).sum(dim=-1), sample_valid
        )

        credit = build_soft_map_credit_targets(
            refined_metrics["horizon_match"].detach(),
            refined_cost.detach(),
            refined_metrics["pair_valid"].detach(),
        )
        raw_horizon_logits = output["raw_horizon_logits"]
        credit_target = credit["credited_target"].type_as(raw_horizon_logits)
        credit_mask = credit["supervision_mask"].type_as(raw_horizon_logits)
        positive_weight = float(
            self.model_cfg.get("CREDIT_POSITIVE_WEIGHT", 6.0)
        )
        credit_bce = F.binary_cross_entropy_with_logits(
            raw_horizon_logits,
            credit_target,
            reduction="none",
        ) * (1.0 + (positive_weight - 1.0) * credit_target)
        loss_credit = (credit_bce * credit_mask).sum() / credit_mask.sum().clamp_min(1)

        valid_weight = (
            pair_valid * self.horizon_weights.type_as(refined)[None]
        )
        weighted_cost = (refined_cost.detach() * valid_weight[:, None]).sum(
            dim=-1
        ) / valid_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        rank_temperature = max(
            float(self.model_cfg.get("RANK_TEMPERATURE", 0.35)), self.eps
        )
        rank_target = torch.softmax(-weighted_cost / rank_temperature, dim=-1)
        loss_rank = -(
            rank_target
            * torch.log_softmax(output["joint_logits"], dim=-1)
        ).sum(dim=-1)
        loss_rank = self._weighted_mean(loss_rank, sample_valid)

        steps = self.measurement_steps.to(base.device)
        base_signature = base.index_select(3, steps).reshape(
            base.shape[0], base.shape[1], -1
        )
        refined_signature = refined.index_select(3, steps).reshape(
            refined.shape[0], refined.shape[1], -1
        )
        base_distance = torch.cdist(
            base_signature.float(), base_signature.float()
        ).type_as(base)
        refined_distance = torch.cdist(
            refined_signature.float(), refined_signature.float()
        ).type_as(base)
        mode_mask = torch.triu(
            torch.ones(
                base.shape[1],
                base.shape[1],
                device=base.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None]
        diversity = F.relu(
            float(self.model_cfg.get("DIVERSITY_RETENTION", 0.90))
            * base_distance
            - refined_distance
        )
        loss_diversity = diversity.masked_select(mode_mask).mean()

        dense_delta = output["dense_delta"]
        delta_velocity = dense_delta[..., 1:, :] - dense_delta[..., :-1, :]
        delta_acceleration = delta_velocity[..., 1:, :] - delta_velocity[..., :-1, :]
        loss_residual = dense_delta.square().mean()
        loss_smooth = delta_acceleration.square().mean()
        loss_score_reg = output["score_delta"].square().mean()

        geometry_total = (
            float(self.model_cfg.get("LOSS_WEIGHT_FLOW", 1.0)) * loss_flow
            + float(self.model_cfg.get("LOSS_WEIGHT_GATE", 0.5)) * loss_gate
            + float(self.model_cfg.get("LOSS_WEIGHT_WINNER_ADE", 0.5))
            * loss_winner_ade
            + float(self.model_cfg.get("LOSS_WEIGHT_COVERAGE", 0.5))
            * loss_coverage
            + float(self.model_cfg.get("LOSS_WEIGHT_NO_HARM", 1.0))
            * loss_no_harm
            + float(self.model_cfg.get("LOSS_WEIGHT_DIVERSITY", 0.1))
            * loss_diversity
            + float(self.model_cfg.get("LOSS_WEIGHT_RESIDUAL", 0.01))
            * loss_residual
            + float(self.model_cfg.get("LOSS_WEIGHT_SMOOTH", 0.02))
            * loss_smooth
        )
        score_total = (
            float(self.model_cfg.get("LOSS_WEIGHT_CREDIT", 1.0))
            * loss_credit
            + float(self.model_cfg.get("LOSS_WEIGHT_RANK", 0.5)) * loss_rank
            + float(self.model_cfg.get("LOSS_WEIGHT_SCORE_REG", 0.02))
            * loss_score_reg
        )
        if loss_mode == "geometry":
            total = geometry_total
        elif loss_mode == "score":
            total = score_total
        elif loss_mode == "joint":
            total = geometry_total + score_total
        else:
            raise ValueError(f"Unknown residual-flow loss mode: {loss_mode}")
        return total, {
            "loss_candidate_residual_flow": total,
            "loss_residual_flow_matching": loss_flow,
            "loss_residual_flow_gate": loss_gate,
            "loss_residual_flow_winner_ade": loss_winner_ade,
            "loss_residual_flow_coverage": loss_coverage,
            "loss_residual_flow_no_harm": loss_no_harm,
            "loss_residual_flow_credit": loss_credit,
            "loss_residual_flow_rank": loss_rank,
            "loss_residual_flow_diversity": loss_diversity,
            "loss_residual_flow_score_reg": loss_score_reg,
            "residual_flow_base_oracle_ade": base_metrics["ade"].amin(
                dim=1
            ).mean(),
            "residual_flow_refined_oracle_ade": refined_metrics["ade"].amin(
                dim=1
            ).mean(),
            "residual_flow_base_coverage": (
                base_metrics["horizon_match"].any(dim=1).type_as(base)
                * pair_valid
            ).sum()
            / pair_valid.sum().clamp_min(1),
            "residual_flow_refined_coverage": (
                refined_metrics["horizon_match"].any(dim=1).type_as(base)
                * pair_valid
            ).sum()
            / pair_valid.sum().clamp_min(1),
            "residual_flow_gain": output["flow_gain"],
            "residual_flow_score_gain": output["score_gain"],
            "residual_flow_delta_rms": dense_delta.square().mean().sqrt(),
            "residual_flow_gate_positive_rate": (
                (output["candidate_gate"] > 0.5).type_as(refined).mean()
            ),
        }
