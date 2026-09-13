"""Core components of Joint Future Exploration and Reasoning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_reasoning import _build_mlp

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
        base_admission_logits = self.admission_head(summary).squeeze(-1)
        selection_score_delta = self.max_score_delta * torch.tanh(
            self.score_head(summary).squeeze(-1)
        )
        base_selection_logits = (
            donor_logits.detach()
            + self.score_bias[None]
            + self.pair_prior_score_weight
            * (
                proposal_prior
                if proposal_prior is not None
                else donor_score.new_zeros(donor_score.shape)
            )
            + selection_score_delta
            + F.logsigmoid(base_admission_logits)
        )
        replacement_utility = base_admission_logits
        candidate_selector_logits = base_admission_logits
        scene_replacement_gate_logits = base_admission_logits.amax(dim=-1)
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
            admission_logits = base_admission_logits
            selection_logits = (
                base_floor
                + self.set_gain_score_scale
                * torch.tanh(admission_logits)
                + 0.0 * base_selection_logits
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
                # Keep the base scoring heads in the DDP graph while the unified
                # utility head owns the deployment decision.
                + 0.0
                * (base_selection_logits + base_admission_logits)
            )
        else:
            admission_logits = base_admission_logits
            selection_logits = base_selection_logits
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
