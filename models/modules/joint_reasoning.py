import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import common as common_utils, loss as loss_utils
from .horizon_scoring import (
    WorldHorizonReliabilityHead,
    build_soft_map_credit_targets,
)


def _build_mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class IntegratedJointWorldDecoder(nn.Module):
    """Pair-conditioned world reasoning inside every base decoder layer.

    The base decoder retains object/map cross-attention. This module binds
    the two target queries into one joint hypothesis before the first layer,
    couples all joint modes after every scene-attention update, and rolls each
    joint trajectory through a scene-conditioned latent world before the next
    layer. The deployment path only sees the learned scene prior; the future
    posterior is used exclusively as a training target.
    """

    def __init__(
        self,
        cfg,
        query_dim,
        map_dim,
        num_future_frames,
        num_decoder_layers,
    ):
        super().__init__()
        self.model_cfg = cfg
        self.query_dim = int(query_dim)
        self.map_dim = int(map_dim)
        self.num_future_frames = int(num_future_frames)
        self.num_decoder_layers = int(num_decoder_layers)
        self.hidden_dim = int(cfg.get("HIDDEN_DIM", 256))
        self.latent_dim = int(cfg.get("LATENT_DIM", 64))
        self.num_heads = int(cfg.get("NUM_HEADS", 8))
        self.num_worlds = int(cfg.get("NUM_WORLDS", 4))
        self.num_output_modes = int(cfg.get("NUM_OUTPUT_MODES", 6))
        self.num_query_modes = int(
            cfg.get("NUM_QUERY_MODES", self.num_output_modes)
        )
        self.max_query_modes = int(cfg.get("MAX_QUERY_MODES", 128))
        self.direct_sparse_modes = bool(
            cfg.get("DIRECT_SPARSE_MODES", False)
        )
        self.ordered_mode_decoding = bool(
            cfg.get("ORDERED_MODE_DECODING", False)
        )
        self.mode_rearrangement = bool(
            cfg.get("MODE_REARRANGEMENT", False)
        )
        self.distinct_intent_assignment = bool(
            cfg.get("DISTINCT_INTENT_ASSIGNMENT", False)
        )
        self.rank_position_encoding = bool(
            cfg.get("RANK_POSITION_ENCODING", False)
        )
        self.consistent_mode_reordering = bool(
            cfg.get("CONSISTENT_MODE_REORDERING", False)
        )
        self.rank_position_scale = float(
            cfg.get("RANK_POSITION_SCALE", 1.0)
        )
        self.use_ma_emta = bool(cfg.get("USE_MA_EMTA", False))
        self.use_progressive_emta = bool(
            cfg.get("USE_PROGRESSIVE_EMTA", False)
        )
        self.use_horizon_listwise = bool(
            cfg.get("USE_HORIZON_LISTWISE", False)
        )
        self.use_world_mode_alignment = bool(
            cfg.get("USE_WORLD_MODE_ALIGNMENT", False)
        )
        self.direct_joint_score = bool(
            cfg.get("DIRECT_JOINT_SCORE", False)
        )
        self.train_stage = str(cfg.get("TRAIN_STAGE", "decoder_joint"))
        self.horizon_reliability_cfg = cfg.get(
            "HORIZON_RELIABILITY", None
        )
        self.use_horizon_reliability = bool(
            self.horizon_reliability_cfg is not None
            and self.horizon_reliability_cfg.get("ENABLED", False)
        )
        self.world_identity_bias = float(
            cfg.get("WORLD_IDENTITY_BIAS", 0.0)
        )
        self.sparse_intent_init_bias = float(
            cfg.get("SPARSE_INTENT_INIT_BIAS", 0.0)
        )
        self.dropout = float(cfg.get("DROPOUT", 0.1))
        self.dt = float(cfg.get("DT", 0.1))
        self.max_acceleration = float(cfg.get("MAX_ACCELERATION", 6.0))
        self.max_rollout_speed = float(cfg.get("MAX_ROLLOUT_SPEED", 50.0))
        self.max_traj_delta = float(cfg.get("MAX_TRAJ_DELTA", 0.75))
        self.max_score_delta = float(cfg.get("MAX_SCORE_DELTA", 0.5))
        self.nms_dist_thresh = float(cfg.get("JOINT_NMS_DIST_THRESH", 2.0))
        self.multihorizon_joint_nms = bool(
            cfg.get("MULTIHORIZON_JOINT_NMS", False)
        )
        self.protected_base_mode_count = int(
            cfg.get("PROTECTED_BASE_MODE_COUNT", 0)
        )
        protected_expansion_cfg = cfg.get("PROTECTED_EXPANSION", {})
        self.protected_expansion_enabled = bool(
            protected_expansion_cfg.get("ENABLED", False)
        )
        self.expand_after_decoder = bool(
            protected_expansion_cfg.get("EXPAND_AFTER_DECODER", False)
        )
        self.expansion_selection_mode = str(
            protected_expansion_cfg.get("SELECTION_MODE", "base_only")
        ).lower()
        self.max_expansion_replacements = int(
            protected_expansion_cfg.get("MAX_REPLACEMENTS", 0)
        )
        self.expansion_admission_threshold = float(
            protected_expansion_cfg.get("ADMISSION_THRESHOLD", 0.5)
        )
        self.expansion_replacement_margin = float(
            protected_expansion_cfg.get("REPLACEMENT_SCORE_MARGIN", 0.0)
        )
        self.expansion_initial_score_bias = float(
            protected_expansion_cfg.get("INITIAL_SCORE_BIAS", -3.0)
        )
        self.eps = 1e-6

        if self.num_output_modes <= 0:
            raise ValueError("NUM_OUTPUT_MODES must be positive")
        if self.direct_sparse_modes:
            if self.num_query_modes < self.num_output_modes:
                raise ValueError(
                    "NUM_QUERY_MODES must be at least NUM_OUTPUT_MODES"
                )
            if self.num_query_modes > self.max_query_modes:
                raise ValueError(
                    "NUM_QUERY_MODES exceeds MAX_QUERY_MODES: "
                    f"{self.num_query_modes} > {self.max_query_modes}"
                )
        if self.distinct_intent_assignment and not self.direct_sparse_modes:
            raise ValueError(
                "DISTINCT_INTENT_ASSIGNMENT requires "
                "DIRECT_SPARSE_MODES=True"
            )
        if self.protected_base_mode_count < 0:
            raise ValueError("PROTECTED_BASE_MODE_COUNT must be non-negative")
        if self.expand_after_decoder:
            self.num_expansion_modes = int(
                protected_expansion_cfg.get("NUM_EXPANSION_MODES", 6)
            )
        else:
            self.num_expansion_modes = max(
                self.num_query_modes - self.protected_base_mode_count, 0
            )
        self.decoder_has_expansion_modes = (
            self.protected_expansion_enabled
            and not self.expand_after_decoder
        )
        if self.protected_expansion_enabled:
            if not self.direct_sparse_modes:
                raise ValueError(
                    "PROTECTED_EXPANSION requires DIRECT_SPARSE_MODES=True"
                )
            if self.protected_base_mode_count != self.num_output_modes:
                raise ValueError(
                    "Protected base mode count must equal NUM_OUTPUT_MODES"
                )
            if self.num_expansion_modes <= 0:
                raise ValueError(
                    "PROTECTED_EXPANSION requires expansion modes"
                )
            if (
                self.expand_after_decoder
                and self.num_query_modes != self.protected_base_mode_count
            ):
                raise ValueError(
                    "Post-decoder expansion requires NUM_QUERY_MODES to "
                    "equal PROTECTED_BASE_MODE_COUNT"
                )
            if self.expansion_selection_mode not in {
                "base_only",
                "guarded_replace",
            }:
                raise ValueError(
                    "PROTECTED_EXPANSION.SELECTION_MODE must be base_only "
                    "or guarded_replace"
                )
            if not 0 <= self.max_expansion_replacements <= self.num_output_modes:
                raise ValueError(
                    "PROTECTED_EXPANSION.MAX_REPLACEMENTS is out of range"
                )
            if not 0.0 <= self.expansion_admission_threshold <= 1.0:
                raise ValueError(
                    "PROTECTED_EXPANSION.ADMISSION_THRESHOLD must be in [0, 1]"
                )

        sample_indices = cfg.get(
            "TEMPORAL_SAMPLE_INDICES", list(range(4, num_future_frames, 5))
        )
        sample_indices = sorted(
            set(
                min(max(int(index), 0), self.num_future_frames - 1)
                for index in sample_indices
            )
        )
        self.register_buffer(
            "sample_indices",
            torch.tensor(sample_indices, dtype=torch.long),
            persistent=False,
        )
        self.num_steps = len(sample_indices)

        self.center_proj = _build_mlp(query_dim, self.hidden_dim, self.hidden_dim)
        self.object_proj = _build_mlp(query_dim, self.hidden_dim, self.hidden_dim)
        self.map_proj = _build_mlp(map_dim, self.hidden_dim, self.hidden_dim)
        self.state_proj = _build_mlp(6, self.hidden_dim, self.hidden_dim)
        self.scene_role_embedding = nn.Parameter(
            torch.zeros(8, self.hidden_dim)
        )
        nn.init.normal_(self.scene_role_embedding, std=0.02)
        self.scene_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(cfg.get("NUM_SCENE_LAYERS", 2)),
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.world_embedding = nn.Parameter(
            torch.empty(self.num_worlds, self.hidden_dim)
        )
        nn.init.normal_(self.world_embedding, std=0.02)
        self.world_slot_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(cfg.get("NUM_WORLD_LAYERS", 2)),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.prior_head = nn.Linear(self.hidden_dim, self.latent_dim * 2)
        self.posterior_head = _build_mlp(
            self.hidden_dim * 2,
            self.hidden_dim * 2,
            self.latent_dim * 2,
        )
        self.latent_to_world = _build_mlp(
            self.latent_dim, self.hidden_dim, self.hidden_dim
        )
        self.world_probability_head = nn.Linear(self.hidden_dim, 1)

        self.future_state_proj = _build_mlp(
            5, self.hidden_dim, self.hidden_dim
        )
        self.future_agent_embedding = nn.Parameter(
            torch.zeros(2, self.hidden_dim)
        )
        self.future_time_embedding = nn.Parameter(
            torch.zeros(self.num_steps, self.hidden_dim)
        )
        nn.init.normal_(self.future_agent_embedding, std=0.02)
        nn.init.normal_(self.future_time_embedding, std=0.02)
        self.future_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(cfg.get("NUM_POSTERIOR_LAYERS", 2)),
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.rollout_state_proj = _build_mlp(
            6, self.hidden_dim, self.hidden_dim
        )
        self.rollout_agent_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.rollout_gru = nn.GRUCell(self.hidden_dim * 3, self.hidden_dim)
        self.rollout_acceleration_head = _build_mlp(
            self.hidden_dim, self.hidden_dim, 2
        )
        self.rollout_time_embedding = nn.Parameter(
            torch.zeros(self.num_steps, self.hidden_dim)
        )
        nn.init.normal_(self.rollout_time_embedding, std=0.02)
        nn.init.zeros_(self.rollout_acceleration_head[-1].weight)
        nn.init.zeros_(self.rollout_acceleration_head[-1].bias)

        self.agent_query_proj = nn.Sequential(
            nn.Linear(query_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.mode_embedding = nn.Parameter(
            torch.empty(self.max_query_modes, self.hidden_dim)
        )
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.rank_position_embedding = (
            nn.Parameter(
                torch.empty(self.max_query_modes, self.hidden_dim)
            )
            if self.rank_position_encoding
            else None
        )
        if self.rank_position_embedding is not None:
            nn.init.normal_(self.rank_position_embedding, std=0.02)
        self.mode_score_bias = nn.Parameter(
            torch.zeros(self.max_query_modes)
        )
        if self.protected_expansion_enabled:
            donor_indices = protected_expansion_cfg.get(
                "DONOR_MODE_INDICES",
                [3, 4, 3, 4, 3, 4],
            )
            if (
                bool(protected_expansion_cfg.get("FULL_BANK_DECODING", False))
                and len(donor_indices) != self.num_expansion_modes
            ):
                # Post-decoder full-bank generation chooses the actual donor
                # per scene from endpoint distance. These cyclic indices only
                # initialize otherwise unused generic expansion parameters.
                donor_indices = [
                    idx % self.protected_base_mode_count
                    for idx in range(self.num_expansion_modes)
                ]
            if len(donor_indices) != self.num_expansion_modes:
                raise ValueError(
                    "PROTECTED_EXPANSION.DONOR_MODE_INDICES must contain "
                    f"{self.num_expansion_modes} entries"
                )
            donor_indices = torch.as_tensor(donor_indices, dtype=torch.long)
            if (
                (donor_indices < 0).any()
                or (donor_indices >= self.protected_base_mode_count).any()
            ):
                raise ValueError(
                    "Expansion donor indices must point into the base bank"
                )
            self.register_buffer(
                "expansion_donor_indices",
                donor_indices,
                persistent=False,
            )
            self.expansion_mode_offset = nn.Parameter(
                torch.zeros(self.num_expansion_modes, self.hidden_dim)
            )
            self.expansion_score_bias = nn.Parameter(
                torch.full(
                    (self.num_expansion_modes,),
                    self.expansion_initial_score_bias,
                )
            )
        else:
            self.register_buffer(
                "expansion_donor_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.expansion_mode_offset = None
            self.expansion_score_bias = None
        self.intent_slot_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intent_identity_bias = float(
            cfg.get("INTENT_IDENTITY_BIAS", 6.0)
        )
        self.intent_temperature = float(
            cfg.get("INTENT_TEMPERATURE", 1.0)
        )
        self.intent_gate_logit = nn.Parameter(
            torch.tensor(float(cfg.get("INTENT_GATE_BIAS_INIT", -2.0)))
        )
        self.joint_seed_fusion = _build_mlp(
            self.hidden_dim * 3,
            self.hidden_dim * 2,
            self.hidden_dim,
            self.dropout,
        )
        self.joint_mode_fusion = nn.ModuleList(
            [
                _build_mlp(
                    self.hidden_dim * 4,
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    self.dropout,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.agent_interaction = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.mode_self_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.scene_cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.joint_ffn = nn.ModuleList(
            [
                _build_mlp(
                    self.hidden_dim,
                    self.hidden_dim * 4,
                    self.hidden_dim,
                    self.dropout,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.agent_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_decoder_layers)]
        )
        self.mode_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_decoder_layers)]
        )
        self.scene_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_decoder_layers)]
        )
        self.ffn_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_decoder_layers)]
        )
        self.query_update_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim * 2, query_dim)
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.initial_query_head = nn.Linear(
            self.hidden_dim, query_dim * 2
        )
        self.expansion_joint_adapters = nn.ModuleList()
        self.expansion_query_adapters = nn.ModuleList()
        if self.decoder_has_expansion_modes:
            for _ in range(self.num_decoder_layers):
                self.expansion_joint_adapters.append(
                    _build_mlp(
                        self.hidden_dim * 3,
                        self.hidden_dim * 2,
                        self.hidden_dim,
                        self.dropout,
                    )
                )
                self.expansion_query_adapters.append(
                    _build_mlp(
                        self.hidden_dim * 3,
                        self.hidden_dim * 2,
                        self.query_dim * 2,
                        self.dropout,
                    )
                )
            self.expansion_adapter_gate_logits = nn.Parameter(
                torch.full(
                    (self.num_decoder_layers,),
                    float(
                        protected_expansion_cfg.get(
                            "ADAPTER_GATE_BIAS", -1.5
                        )
                    ),
                )
            )
        else:
            self.expansion_adapter_gate_logits = None

        candidate_feature_dim = 14
        self.candidate_state_proj = _build_mlp(
            candidate_feature_dim, self.hidden_dim, self.hidden_dim
        )
        self.candidate_agent_embedding = nn.Parameter(
            torch.zeros(2, self.hidden_dim)
        )
        self.candidate_time_embedding = nn.Parameter(
            torch.zeros(self.num_steps, self.hidden_dim)
        )
        nn.init.normal_(self.candidate_agent_embedding, std=0.02)
        nn.init.normal_(self.candidate_time_embedding, std=0.02)
        self.candidate_temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.world_response_fusion = nn.ModuleList(
            [
                _build_mlp(
                    self.hidden_dim * 4,
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    self.dropout,
                )
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.feedback_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, query_dim * 2)
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.score_heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_decoder_layers)]
        )
        self.traj_control_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.num_steps * 2)
                for _ in range(self.num_decoder_layers)
            ]
        )
        self.query_gate_logits = nn.Parameter(
            torch.full(
                (self.num_decoder_layers,),
                float(cfg.get("QUERY_GATE_BIAS_INIT", -2.0)),
            )
        )
        self.feedback_gate_logits = nn.Parameter(
            torch.full(
                (self.num_decoder_layers,),
                float(cfg.get("FEEDBACK_GATE_BIAS_INIT", -2.0)),
            )
        )
        self.score_gate_logits = nn.Parameter(
            torch.full(
                (self.num_decoder_layers,),
                float(cfg.get("SCORE_GATE_BIAS_INIT", -2.0)),
            )
        )
        self.traj_gate_logits = nn.Parameter(
            torch.full(
                (self.num_decoder_layers,),
                float(cfg.get("TRAJ_GATE_BIAS_INIT", -2.0)),
            )
        )
        self.initial_query_gate = nn.Parameter(
            torch.tensor(float(cfg.get("INITIAL_QUERY_GATE_BIAS", -2.0)))
        )
        self.world_reliability_head = (
            WorldHorizonReliabilityHead(
                cfg=self.horizon_reliability_cfg,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
            )
            if self.use_horizon_reliability
            else None
        )
        self._initialize_residual_heads()
        if self.decoder_has_expansion_modes:
            for head in self.expansion_joint_adapters:
                nn.init.normal_(head[-1].weight, std=1e-3)
                nn.init.zeros_(head[-1].bias)
            for head in self.expansion_query_adapters:
                nn.init.normal_(head[-1].weight, std=1e-3)
                nn.init.zeros_(head[-1].bias)

    def train_horizon_reliability_modules(self):
        if self.world_reliability_head is None:
            raise RuntimeError(
                "horizon_reliability_warmup requires "
                "HORIZON_RELIABILITY.ENABLED=True"
            )
        # Keep candidate generation on the exact deployment path. In
        # particular, do not encode the future-conditioned posterior merely
        # because the reliability head is being optimized.
        self.training = False
        self.world_reliability_head.train(True)

    def _initialize_residual_heads(self):
        for head in self.query_update_heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.normal_(self.initial_query_head.weight, std=1e-3)
        nn.init.zeros_(self.initial_query_head.bias)
        for heads in (self.feedback_heads, self.traj_control_heads):
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        for head in self.score_heads:
            if self.direct_joint_score:
                nn.init.normal_(head.weight, std=0.02)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _masked_mean(features, mask, dim):
        weight = mask.type_as(features)
        while weight.dim() < features.dim():
            weight = weight.unsqueeze(-1)
        return (features * weight).sum(dim=dim) / weight.sum(
            dim=dim
        ).clamp_min(1.0)

    def _reshape_pair(self, tensor, batch_size):
        if tensor.shape[0] != batch_size * 2:
            raise RuntimeError(
                "Integrated joint decoding requires exactly two center views "
                f"per scenario, got {tensor.shape[0]} centers for {batch_size} pairs"
            )
        return tensor.reshape(batch_size, 2, *tensor.shape[1:])

    def _encode_future(self, input_dict, dtype, device):
        gt = input_dict["pair_gt_trajs"].to(device).type(dtype)[..., :4]
        mask = input_dict["pair_gt_trajs_mask"].to(device).bool()
        sample_idx = self.sample_indices.to(device)
        sampled_gt = gt.index_select(2, sample_idx)
        sampled_mask = mask.index_select(2, sample_idx)
        feature = torch.cat(
            [
                sampled_gt
                / sampled_gt.new_tensor([50.0, 50.0, 20.0, 20.0]),
                sampled_mask[..., None].type_as(sampled_gt),
            ],
            dim=-1,
        )
        token = self.future_state_proj(feature)
        token = (
            token
            + self.future_agent_embedding[None, :, None]
            + self.future_time_embedding[None, None]
        )
        batch_size = token.shape[0]
        token = token.reshape(batch_size, 2 * self.num_steps, self.hidden_dim)
        flat_mask = sampled_mask.reshape(batch_size, 2 * self.num_steps)
        safe_mask = flat_mask.clone()
        no_future = ~safe_mask.any(dim=-1)
        if no_future.any():
            safe_mask[no_future, 0] = True
            token = token.clone()
            token[no_future, 0] = 0.0
        token = self.future_encoder(token, src_key_padding_mask=~safe_mask)
        pooled = self._masked_mean(token, safe_mask, dim=1)
        return pooled, sampled_gt, sampled_mask

    def _rollout_worlds(self, world_token, scene_token, pair_state):
        batch_size, num_worlds, _ = world_token.shape
        hidden = self.rollout_state_proj(pair_state)[:, None]
        hidden = hidden + world_token[:, :, None]
        hidden = hidden.reshape(
            batch_size * num_worlds, 2, self.hidden_dim
        )
        world_context = world_token[:, :, None].expand(
            -1, -1, 2, -1
        ).reshape(batch_size * num_worlds, 2, self.hidden_dim)
        scene_context = scene_token[:, None, None].expand(
            -1, num_worlds, 2, -1
        ).reshape(batch_size * num_worlds, 2, self.hidden_dim)
        position = pair_state[..., 0:2][:, None].expand(
            -1, num_worlds, -1, -1
        ).reshape(batch_size * num_worlds, 2, 2)
        velocity = pair_state[..., 2:4][:, None].expand(
            -1, num_worlds, -1, -1
        ).reshape(batch_size * num_worlds, 2, 2)

        rollout = []
        previous_index = -1
        for step_idx, sample_index in enumerate(self.sample_indices.tolist()):
            interacted, _ = self.rollout_agent_attention(
                hidden, hidden, hidden, need_weights=False
            )
            interacted = interacted + self.rollout_time_embedding[
                step_idx
            ].view(1, 1, -1)
            gru_input = torch.cat(
                [interacted, scene_context, world_context], dim=-1
            )
            hidden = self.rollout_gru(
                gru_input.reshape(-1, self.hidden_dim * 3),
                hidden.reshape(-1, self.hidden_dim),
            ).reshape(batch_size * num_worlds, 2, self.hidden_dim)
            delta_t = max((sample_index - previous_index) * self.dt, self.dt)
            acceleration = (
                torch.tanh(self.rollout_acceleration_head(hidden))
                * self.max_acceleration
            )
            next_velocity = velocity + acceleration * delta_t
            if self.max_rollout_speed > 0.0:
                speed = torch.linalg.vector_norm(
                    next_velocity, dim=-1, keepdim=True
                ).clamp_min(self.eps)
                next_velocity = next_velocity * (
                    self.max_rollout_speed / speed
                ).clamp(max=1.0)
            position = position + 0.5 * (velocity + next_velocity) * delta_t
            velocity = next_velocity
            rollout.append(torch.cat([position, velocity], dim=-1))
            previous_index = sample_index
        rollout = torch.stack(rollout, dim=2)
        return rollout.reshape(
            batch_size, num_worlds, 2, self.num_steps, 4
        )

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
        pair_state = input_dict["pair_current_state"].to(
            center_feature.device
        ).type_as(center_feature)
        batch_size = pair_state.shape[0]
        center_pair = self._reshape_pair(center_feature, batch_size)
        obj_pair = self._reshape_pair(obj_feature, batch_size)
        obj_mask_pair = self._reshape_pair(obj_mask, batch_size)
        map_pair = self._reshape_pair(map_feature, batch_size)
        map_mask_pair = self._reshape_pair(map_mask, batch_size)

        obj_pool = self._masked_mean(obj_pair, obj_mask_pair, dim=2)
        map_pool = self._masked_mean(map_pair, map_mask_pair, dim=2)
        scene_tokens = torch.cat(
            [
                self.center_proj(center_pair),
                self.object_proj(obj_pool),
                self.map_proj(map_pool),
                self.state_proj(pair_state),
            ],
            dim=1,
        )
        scene_tokens = scene_tokens + self.scene_role_embedding[None]
        scene_memory = self.scene_encoder(scene_tokens)
        scene_token = scene_memory.mean(dim=1)

        world_slots = self.world_embedding[None] + scene_token[:, None]
        world_slots = self.world_slot_encoder(world_slots)
        prior_mu, prior_logvar = self.prior_head(world_slots).chunk(2, dim=-1)
        prior_logvar = prior_logvar.clamp(-6.0, 4.0)
        prior_world_token = world_slots + self.latent_to_world(prior_mu)
        prior_rollout = self._rollout_worlds(
            prior_world_token, scene_token, pair_state
        )

        posterior_mu = posterior_logvar = posterior_world_token = None
        posterior_rollout = sampled_gt = sampled_mask = None
        if self.training and "pair_gt_trajs" in input_dict:
            future_token, sampled_gt, sampled_mask = self._encode_future(
                input_dict, center_feature.dtype, center_feature.device
            )
            posterior_mu, posterior_logvar = self.posterior_head(
                torch.cat(
                    [
                        world_slots,
                        future_token[:, None].expand_as(world_slots),
                    ],
                    dim=-1,
                )
            ).chunk(2, dim=-1)
            posterior_logvar = posterior_logvar.clamp(-6.0, 4.0)
            posterior_world_token = world_slots + self.latent_to_world(
                posterior_mu
            )
            posterior_rollout = self._rollout_worlds(
                posterior_world_token, scene_token, pair_state
            )

        world_logits = self.world_probability_head(world_slots).squeeze(-1)
        memory = torch.cat([scene_memory, prior_world_token], dim=1)
        return {
            "batch_size": batch_size,
            "pair_state": pair_state,
            "scene_memory": scene_memory,
            "scene_token": scene_token,
            "memory": memory,
            "world_slots": world_slots,
            "world_token": prior_world_token,
            "world_logits": world_logits,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "prior_rollout": prior_rollout,
            "posterior_mu": posterior_mu,
            "posterior_logvar": posterior_logvar,
            "posterior_world_token": posterior_world_token,
            "posterior_rollout": posterior_rollout,
            "sampled_gt": sampled_gt,
            "sampled_mask": sampled_mask,
        }

    def _assign_worlds(self, joint_token, state):
        logits = torch.einsum(
            "bkh,bwh->bkw", joint_token, state["world_token"]
        ) / math.sqrt(self.hidden_dim)
        logits = logits + state["world_logits"][:, None]
        if self.world_identity_bias > 0.0:
            if logits.shape[1] == logits.shape[2]:
                identity = torch.eye(
                    logits.shape[1],
                    device=logits.device,
                    dtype=logits.dtype,
                )[None]
                logits = logits + self.world_identity_bias * identity
            elif (
                self.direct_sparse_modes
                and logits.shape[1] > logits.shape[2]
            ):
                # Preserve the original one-mode/one-world assignment for the
                # first bank and give each expansion mode a stable cyclic
                # world prior. Learned attention can still override the bias.
                world_index = torch.arange(
                    logits.shape[1], device=logits.device
                ) % logits.shape[2]
                identity = F.one_hot(
                    world_index, num_classes=logits.shape[2]
                ).type_as(logits)[None]
                logits = logits + self.world_identity_bias * identity
        assignment = torch.softmax(logits, dim=-1)
        attended_world = torch.einsum(
            "bkw,bwh->bkh", assignment, state["world_token"]
        )
        attended_rollout = torch.einsum(
            "bkw,bwash->bkash", assignment, state["prior_rollout"]
        )
        return assignment, attended_world, attended_rollout

    def _mode_slot_seed(self, scene, num_modes):
        """Build query slots while keeping the loaded base bank immutable."""
        if not self.decoder_has_expansion_modes:
            return self.mode_embedding[None, :num_modes] + scene

        protected = self.protected_base_mode_count
        if num_modes != protected + self.num_expansion_modes:
            raise RuntimeError(
                "Protected expansion received an unexpected mode count: "
                f"{num_modes}"
            )
        base_seed = self.mode_embedding[:protected]
        donor_seed = self.mode_embedding.index_select(
            0, self.expansion_donor_indices
        ).detach()
        expansion_seed = donor_seed + self.expansion_mode_offset
        seed = torch.cat([base_seed, expansion_seed], dim=0)
        return seed[None] + scene

    def _joint_mode_score_bias(self, num_modes):
        if not self.protected_expansion_enabled:
            return self.mode_score_bias[:num_modes]
        protected = self.protected_base_mode_count
        if self.expand_after_decoder and num_modes == protected:
            return self.mode_score_bias[:protected]
        if num_modes != protected + self.num_expansion_modes:
            raise RuntimeError(
                "Protected score prior received an unexpected mode count"
            )
        return torch.cat(
            [self.mode_score_bias[:protected], self.expansion_score_bias],
            dim=0,
        )

    def _sparse_intent_seed_indices(
        self, agent_intent_points, num_modes, pair_state
    ):
        batch_size, num_intents, num_agents, _ = agent_intent_points.shape
        points = agent_intent_points.permute(0, 2, 1, 3)
        radii = torch.linalg.vector_norm(points, dim=-1)
        speed = torch.linalg.vector_norm(
            pair_state[..., 2:4], dim=-1
        )
        lower_radius = torch.quantile(radii, 0.2, dim=-1)
        upper_radius = torch.quantile(radii, 0.8, dim=-1)
        moving_radius = torch.maximum(speed * 8.0, lower_radius)
        moving_radius = torch.minimum(moving_radius, upper_radius)

        if num_modes in {6, 12} and num_agents == 2:
            # The first six are the stable Graph-v6 bank. Expansion adds six
            # asymmetric turn combinations that the coupled six-mode bank
            # cannot represent, while preserving the old seeds exactly.
            radius_scale = points.new_tensor(
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            )[:num_modes]
            angle = points.new_tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.65, 0.65],
                    [-0.65, -0.65],
                    [0.0, 0.0],
                    [0.65, 0.0],
                    [-0.65, 0.0],
                    [0.0, 0.65],
                    [0.0, -0.65],
                    [0.65, -0.65],
                    [-0.65, 0.65],
                ]
            )[:num_modes]
        else:
            angle_1d = torch.linspace(
                -math.pi,
                math.pi,
                num_modes + 1,
                device=points.device,
                dtype=points.dtype,
            )[:-1]
            angle = angle_1d[:, None].expand(-1, num_agents)
            radius_scale = torch.ones_like(angle)

        prototype_radius = (
            moving_radius[:, None] * radius_scale[None]
        )
        prototypes = torch.stack(
            [angle.cos(), angle.sin()], dim=-1
        )[None] * prototype_radius[..., None]
        distance = torch.linalg.vector_norm(
            prototypes[..., None, :] - points[:, None], dim=-1
        )
        return distance.argmin(dim=-1)

    def initialize_queries(self, intention_query, intention_points, state):
        num_intents, num_centers, _ = intention_query.shape
        batch_size = state["batch_size"]
        if num_intents > self.max_query_modes:
            raise RuntimeError(
                f"Got {num_intents} motion queries, MAX_QUERY_MODES is "
                f"{self.max_query_modes}"
            )
        num_modes = (
            self.num_query_modes
            if self.direct_sparse_modes
            else num_intents
        )
        agent_intent_query = intention_query.permute(1, 0, 2).reshape(
            batch_size, 2, num_intents, self.query_dim
        ).permute(0, 2, 1, 3)
        agent_intent_points = intention_points.permute(1, 0, 2).reshape(
            batch_size, 2, num_intents, 2
        ).permute(0, 2, 1, 3)
        agent_intent_hidden = self.agent_query_proj(agent_intent_query)
        scene = state["scene_token"][:, None].expand(-1, num_modes, -1)
        slot_seed = self._mode_slot_seed(scene, num_modes)
        _, slot_world, _ = self._assign_worlds(slot_seed, state)
        slot_query = self.intent_slot_proj(slot_seed + slot_world)
        intent_key = self.intent_key_proj(agent_intent_hidden)
        intent_logits = torch.einsum(
            "bqh,bkah->bqak", slot_query, intent_key
        ) / math.sqrt(self.hidden_dim)
        if self.direct_sparse_modes and self.sparse_intent_init_bias > 0.0:
            seed_idx = self._sparse_intent_seed_indices(
                agent_intent_points, num_modes, state["pair_state"]
            )
            seed_bias = F.one_hot(
                seed_idx, num_classes=num_intents
            ).type_as(intent_logits)
            intent_logits = (
                intent_logits + self.sparse_intent_init_bias * seed_bias
            )
        if not self.direct_sparse_modes and num_modes == intent_logits.shape[-1]:
            identity = torch.eye(
                num_modes,
                device=intent_logits.device,
                dtype=intent_logits.dtype,
            ).view(1, num_modes, 1, num_modes)
            intent_logits = (
                intent_logits + self.intent_identity_bias * identity
            )
        soft_intent_assignment = torch.softmax(
            intent_logits / max(self.intent_temperature, self.eps), dim=-1
        )
        if self.distinct_intent_assignment:
            hard_indices = self._greedy_unique_intent_indices(intent_logits)
            hard_assignment = F.one_hot(
                hard_indices, num_classes=num_intents
            ).type_as(soft_intent_assignment)
            # The forward pass uses actual k-means anchors, while gradients
            # still train the scene-conditioned slot-to-intent assignment.
            intent_assignment = (
                hard_assignment
                + soft_intent_assignment
                - soft_intent_assignment.detach()
            )
        else:
            intent_assignment = soft_intent_assignment
        proposed_query = torch.einsum(
            "bqak,bkad->bqad", intent_assignment, agent_intent_query
        )
        proposed_points = torch.einsum(
            "bqak,bkad->bqad", intent_assignment, agent_intent_points
        )
        if self.direct_sparse_modes:
            agent_query = proposed_query
            agent_points = proposed_points
        else:
            intent_gate = torch.sigmoid(self.intent_gate_logit)
            agent_query = agent_intent_query + intent_gate * (
                proposed_query - agent_intent_query
            )
            agent_points = agent_intent_points + intent_gate * (
                proposed_points - agent_intent_points
            )
        agent_hidden = self.agent_query_proj(agent_query)
        joint_token = self.joint_seed_fusion(
            torch.cat(
                [agent_hidden[:, :, 0], agent_hidden[:, :, 1], scene],
                dim=-1,
            )
        )
        joint_token = joint_token + slot_seed
        assignment, attended_world, _ = self._assign_worlds(
            joint_token, state
        )
        joint_token = joint_token + attended_world
        initial_delta = self.initial_query_head(joint_token).reshape(
            batch_size, num_modes, 2, self.query_dim
        )
        query_content = torch.sigmoid(self.initial_query_gate) * initial_delta
        query_content = query_content.permute(0, 2, 1, 3).reshape(
            num_centers, num_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        state["joint_token"] = joint_token
        state["assignment"] = assignment
        state["intent_assignment"] = intent_assignment
        adapted_intention_query = agent_query.permute(0, 2, 1, 3).reshape(
            num_centers, num_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        adapted_intention_points = agent_points.permute(0, 2, 1, 3).reshape(
            num_centers, num_modes, 2
        ).permute(1, 0, 2).contiguous()
        return (
            adapted_intention_query,
            adapted_intention_points,
            query_content,
            state,
        )

    @staticmethod
    def _gather_joint_modes(tensor, order):
        index = order
        for _ in range(tensor.dim() - 2):
            index = index.unsqueeze(-1)
        index = index.expand(-1, -1, *tensor.shape[2:])
        return tensor.gather(1, index)

    def _gather_center_modes(self, tensor, order):
        batch_size, num_modes = order.shape
        center_view = tensor.reshape(
            batch_size, 2, num_modes, *tensor.shape[2:]
        ).transpose(1, 2)
        center_view = self._gather_joint_modes(center_view, order)
        return center_view.transpose(1, 2).reshape(
            batch_size * 2, num_modes, *tensor.shape[2:]
        )

    def _gather_query_modes(self, query_content, order):
        center_view = query_content.permute(1, 0, 2).contiguous()
        center_view = self._gather_center_modes(center_view, order)
        return center_view.permute(1, 0, 2).contiguous()

    @staticmethod
    def _greedy_unique_intent_indices(intent_logits):
        """Assign each ordered slot a different anchor for each target.

        Modes are consumed in causal order, so an earlier mode keeps its
        highest-scoring anchor and later modes choose the best unused anchor.
        This prevents a direct sparse mode from becoming a convex average of
        incompatible k-means intentions.
        """
        batch_size, num_modes, num_agents, num_intents = (
            intent_logits.shape
        )
        if num_modes > num_intents:
            raise ValueError(
                "Distinct intent assignment needs at least one anchor per "
                f"mode, got {num_modes} modes and {num_intents} anchors"
            )
        scores = intent_logits.detach().permute(0, 2, 1, 3)
        used = torch.zeros(
            batch_size,
            num_agents,
            num_intents,
            device=intent_logits.device,
            dtype=torch.bool,
        )
        selected = []
        for mode_idx in range(num_modes):
            available_scores = scores[:, :, mode_idx].masked_fill(
                used, -torch.inf
            )
            anchor_idx = available_scores.argmax(dim=-1)
            selected.append(anchor_idx)
            used.scatter_(2, anchor_idx.unsqueeze(-1), True)
        return torch.stack(selected, dim=1)

    def reorder_query_anchors(
        self, intention_query, intention_points, order
    ):
        """Keep static query embeddings aligned with score-sorted modes."""
        if not self.consistent_mode_reordering:
            return intention_query, intention_points
        return (
            self._gather_query_modes(intention_query, order),
            self._gather_query_modes(intention_points, order),
        )

    def _causal_mode_mask(self, num_modes, device):
        mask = None
        if self.ordered_mode_decoding:
            mask = torch.triu(
                torch.ones(
                    num_modes,
                    num_modes,
                    device=device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        if self.decoder_has_expansion_modes:
            if mask is None:
                mask = torch.zeros(
                    num_modes,
                    num_modes,
                    device=device,
                    dtype=torch.bool,
                )
            protected = min(self.protected_base_mode_count, num_modes)
            mask[:protected, protected:] = True
        return mask

    def _rank_mode_order(self, logits):
        """Rank expanded modes without allowing them to disturb the base bank."""
        num_modes = logits.shape[1]
        protected = min(self.protected_base_mode_count, num_modes)
        if protected <= 0 or protected >= num_modes:
            return logits.detach().argsort(dim=-1, descending=True)
        base_order = logits[:, :protected].detach().argsort(
            dim=-1, descending=True
        )
        expansion_order = logits[:, protected:].detach().argsort(
            dim=-1, descending=True
        ) + protected
        return torch.cat([base_order, expansion_order], dim=-1)

    def couple_queries(self, layer_idx, query_content, state):
        num_modes, num_centers, _ = query_content.shape
        batch_size = state["batch_size"]
        agent_query = query_content.permute(1, 0, 2).reshape(
            batch_size, 2, num_modes, self.query_dim
        ).permute(0, 2, 1, 3)
        agent_hidden = self.agent_query_proj(agent_query)
        flat_agent = agent_hidden.reshape(
            batch_size * num_modes, 2, self.hidden_dim
        )
        interacted, _ = self.agent_interaction[layer_idx](
            flat_agent, flat_agent, flat_agent, need_weights=False
        )
        agent_hidden = self.agent_norm[layer_idx](
            flat_agent + interacted
        ).reshape(batch_size, num_modes, 2, self.hidden_dim)
        scene = state["scene_token"][:, None].expand(-1, num_modes, -1)
        joint_token = self.joint_mode_fusion[layer_idx](
            torch.cat(
                [
                    agent_hidden[:, :, 0],
                    agent_hidden[:, :, 1],
                    state["joint_token"],
                    scene,
                ],
                dim=-1,
            )
        )
        if self.rank_position_embedding is not None:
            joint_token = (
                joint_token
                + self.rank_position_scale
                * self.rank_position_embedding[None, :num_modes]
            )
        mode_update, _ = self.mode_self_attention[layer_idx](
            joint_token,
            joint_token,
            joint_token,
            attn_mask=self._causal_mode_mask(
                num_modes, joint_token.device
            ),
            need_weights=False,
        )
        joint_token = self.mode_norm[layer_idx](joint_token + mode_update)
        scene_update, _ = self.scene_cross_attention[layer_idx](
            joint_token,
            state["memory"],
            state["memory"],
            need_weights=False,
        )
        joint_token = self.scene_norm[layer_idx](joint_token + scene_update)
        joint_token = self.ffn_norm[layer_idx](
            joint_token + self.joint_ffn[layer_idx](joint_token)
        )
        expansion_query_delta = None
        if self.decoder_has_expansion_modes:
            protected = self.protected_base_mode_count
            _, expansion_world, _ = self._assign_worlds(
                joint_token[:, protected:], state
            )
            expansion_scene = scene[:, protected:]
            expansion_context = torch.cat(
                [
                    joint_token[:, protected:],
                    expansion_world,
                    expansion_scene,
                ],
                dim=-1,
            )
            expansion_gate = torch.sigmoid(
                self.expansion_adapter_gate_logits[layer_idx]
            )
            expansion_joint_delta = self.expansion_joint_adapters[layer_idx](
                expansion_context
            )
            joint_token = torch.cat(
                [
                    joint_token[:, :protected],
                    joint_token[:, protected:]
                    + expansion_gate * expansion_joint_delta,
                ],
                dim=1,
            )
            expansion_query_delta = self.expansion_query_adapters[layer_idx](
                expansion_context
            ).reshape(
                batch_size,
                self.num_expansion_modes,
                2,
                self.query_dim,
            )
        joint_for_agent = joint_token[:, :, None].expand(-1, -1, 2, -1)
        query_delta = self.query_update_heads[layer_idx](
            torch.cat([agent_hidden, joint_for_agent], dim=-1)
        )
        query_gate = torch.sigmoid(self.query_gate_logits[layer_idx])
        agent_query = agent_query + query_gate * query_delta
        if expansion_query_delta is not None:
            protected = self.protected_base_mode_count
            expansion_gate = torch.sigmoid(
                self.expansion_adapter_gate_logits[layer_idx]
            )
            agent_query = torch.cat(
                [
                    agent_query[:, :protected],
                    agent_query[:, protected:]
                    + expansion_gate * expansion_query_delta,
                ],
                dim=1,
            )
        query_content = agent_query.permute(0, 2, 1, 3).reshape(
            num_centers, num_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        assignment, _, _ = self._assign_worlds(joint_token, state)
        state["joint_token"] = joint_token
        state["assignment"] = assignment
        return query_content, state

    @staticmethod
    def _agent1_to_anchor0(agent1_trajs, pair_center_world):
        batch_size, num_modes, num_steps, _ = agent1_trajs.shape
        agent0_world = pair_center_world[:, 0]
        agent1_world = pair_center_world[:, 1]
        world_xy = common_utils.rotate_points_along_z(
            agent1_trajs.reshape(batch_size, num_modes * num_steps, 2),
            agent1_world[:, 6],
        ).reshape(batch_size, num_modes, num_steps, 2)
        world_xy = world_xy + agent1_world[:, None, None, 0:2]
        return common_utils.rotate_points_along_z(
            (world_xy - agent0_world[:, None, None, 0:2]).reshape(
                batch_size, num_modes * num_steps, 2
            ),
            -agent0_world[:, 6],
        ).reshape(batch_size, num_modes, num_steps, 2)

    @staticmethod
    def _anchor0_to_agent1(anchor0_trajs, pair_center_world):
        batch_size, num_modes, num_steps, _ = anchor0_trajs.shape
        agent0_world = pair_center_world[:, 0]
        agent1_world = pair_center_world[:, 1]
        world_xy = common_utils.rotate_points_along_z(
            anchor0_trajs.reshape(batch_size, num_modes * num_steps, 2),
            agent0_world[:, 6],
        ).reshape(batch_size, num_modes, num_steps, 2)
        world_xy = world_xy + agent0_world[:, None, None, 0:2]
        return common_utils.rotate_points_along_z(
            (world_xy - agent1_world[:, None, None, 0:2]).reshape(
                batch_size, num_modes * num_steps, 2
            ),
            -agent1_world[:, 6],
        ).reshape(batch_size, num_modes, num_steps, 2)

    def _to_canonical(self, pred_trajs, pair_center_world, batch_size):
        num_modes = pred_trajs.shape[1]
        local = pred_trajs.reshape(
            batch_size, 2, num_modes, self.num_future_frames, -1
        )
        agent0 = local[:, 0, ..., 0:2]
        agent1 = self._agent1_to_anchor0(
            local[:, 1, ..., 0:2], pair_center_world
        )
        return torch.stack([agent0, agent1], dim=2)

    def _to_local(self, canonical_xy, pred_trajs, pair_center_world):
        batch_size, num_modes = canonical_xy.shape[:2]
        local = pred_trajs.reshape(
            batch_size, 2, num_modes, self.num_future_frames, -1
        ).clone()
        original_xy = local[..., 0:2].clone()
        local[:, 0, ..., 0:2] = canonical_xy[:, :, 0]
        local[:, 1, ..., 0:2] = self._anchor0_to_agent1(
            canonical_xy[:, :, 1], pair_center_world
        )
        if local.shape[-1] >= 7:
            local_delta = local[..., 0:2] - original_xy
            velocity_delta = torch.zeros_like(local_delta)
            velocity_delta[..., 1:, :] = (
                local_delta[..., 1:, :] - local_delta[..., :-1, :]
            ) / self.dt
            velocity_delta[..., 0, :] = velocity_delta[..., 1, :]
            local[..., 5:7] = local[..., 5:7] + velocity_delta
        return local.reshape(
            batch_size * 2,
            num_modes,
            self.num_future_frames,
            local.shape[-1],
        )

    def _candidate_features(self, canonical_xy, attended_rollout):
        sample_idx = self.sample_indices.to(canonical_xy.device)
        sampled_xy = canonical_xy.index_select(3, sample_idx)
        velocity = torch.zeros_like(canonical_xy)
        velocity[..., 1:, :] = (
            canonical_xy[..., 1:, :] - canonical_xy[..., :-1, :]
        ) / self.dt
        velocity[..., 0, :] = velocity[..., 1, :]
        sampled_velocity = velocity.index_select(3, sample_idx)
        world_xy = attended_rollout[..., 0:2]
        world_velocity = attended_rollout[..., 2:4]
        partner_xy = sampled_xy.flip(dims=[2])
        partner_velocity = sampled_velocity.flip(dims=[2])
        relative_xy = partner_xy - sampled_xy
        relative_velocity = partner_velocity - sampled_velocity
        distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
        time = (sample_idx.type_as(canonical_xy) + 1.0) / float(
            self.num_future_frames
        )
        time = time.view(1, 1, 1, self.num_steps, 1).expand(
            canonical_xy.shape[0], canonical_xy.shape[1], 2, -1, -1
        )
        return torch.cat(
            [
                sampled_xy / 50.0,
                sampled_velocity / 20.0,
                (sampled_xy - world_xy) / 30.0,
                (sampled_velocity - world_velocity) / 20.0,
                relative_xy / 30.0,
                relative_velocity / 20.0,
                distance / 20.0,
                time,
            ],
            dim=-1,
        )

    def _interpolate_control(self, control):
        batch_size, num_modes = control.shape[:2]
        full = F.interpolate(
            control.reshape(
                batch_size * num_modes * 2, self.num_steps, 2
            ).transpose(1, 2),
            size=self.num_future_frames,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        progress = torch.linspace(
            1.0 / self.num_future_frames,
            1.0,
            self.num_future_frames,
            device=control.device,
            dtype=control.dtype,
        )
        full = full * progress[None, :, None]
        return full.reshape(
            batch_size, num_modes, 2, self.num_future_frames, 2
        )

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
        pair_center_world = input_dict["pair_center_objects_world"].to(
            pred_trajs.device
        ).type_as(pred_trajs)
        canonical_xy = self._to_canonical(
            pred_trajs, pair_center_world, batch_size
        )
        assignment, attended_world, attended_rollout = self._assign_worlds(
            state["joint_token"], state
        )
        candidate_feature = self._candidate_features(
            canonical_xy, attended_rollout
        )
        token = self.candidate_state_proj(candidate_feature)
        token = (
            token
            + self.candidate_agent_embedding[None, None, :, None]
            + self.candidate_time_embedding[None, None, None]
        )
        token = token.reshape(
            batch_size * num_modes,
            2 * self.num_steps,
            self.hidden_dim,
        )
        token = self.candidate_temporal_layers[layer_idx](token)
        temporal_hidden = token.reshape(
            batch_size, num_modes, 2, self.num_steps, self.hidden_dim
        )
        temporal_summary = temporal_hidden.mean(dim=(2, 3))
        scene = state["scene_token"][:, None].expand(-1, num_modes, -1)
        response = self.world_response_fusion[layer_idx](
            torch.cat(
                [
                    state["joint_token"],
                    temporal_summary,
                    attended_world,
                    scene,
                ],
                dim=-1,
            )
        )

        control = torch.tanh(
            self.traj_control_heads[layer_idx](response)
        ).reshape(batch_size, num_modes, 2, self.num_steps, 2)
        delta = self._interpolate_control(control) * self.max_traj_delta
        traj_gate = torch.sigmoid(self.traj_gate_logits[layer_idx])
        applied_delta = traj_gate * delta
        refined_xy = canonical_xy + applied_delta
        refined_local = self._to_local(
            refined_xy, pred_trajs, pair_center_world
        )

        marginal_logits = pred_scores.reshape(
            batch_size, 2, num_modes
        ).mean(dim=1)
        score_residual = self.score_heads[layer_idx](response).squeeze(-1)
        if self.direct_joint_score:
            score_gate = score_residual.new_ones(())
            applied_score = score_residual
            joint_logits = (
                score_residual
                + self._joint_mode_score_bias(num_modes)[None]
            )
        else:
            score_gate = torch.sigmoid(self.score_gate_logits[layer_idx])
            applied_score = score_gate * score_residual
            if self.max_score_delta > 0.0:
                applied_score = applied_score.clamp(
                    -self.max_score_delta, self.max_score_delta
                )
            joint_logits = marginal_logits + applied_score

        base_joint_logits = joint_logits
        reliability_ret = None
        if (
            self.world_reliability_head is not None
            and layer_idx + 1 == self.num_decoder_layers
        ):
            reliability_ret = self.world_reliability_head(
                world_response=response,
                base_logits=base_joint_logits,
            )
            joint_logits = reliability_ret["fused_logits"]

        feedback = self.feedback_heads[layer_idx](response).reshape(
            batch_size, num_modes, 2, self.query_dim
        )
        feedback_gate = torch.sigmoid(self.feedback_gate_logits[layer_idx])
        agent_query = query_content.permute(1, 0, 2).reshape(
            batch_size, 2, num_modes, self.query_dim
        ).permute(0, 2, 1, 3)
        agent_query = agent_query + feedback_gate * feedback
        query_content = agent_query.permute(0, 2, 1, 3).reshape(
            batch_size * 2, num_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()

        next_joint_token = state["joint_token"] + response
        next_query_content = query_content
        next_agent_scores = pred_scores
        next_agent_trajs = refined_local
        next_intent_assignment = state["intent_assignment"]
        mode_order = torch.arange(
            num_modes, device=joint_logits.device
        )[None].expand(batch_size, -1)
        if self.mode_rearrangement and layer_idx + 1 < self.num_decoder_layers:
            mode_order = self._rank_mode_order(joint_logits)
            next_joint_token = self._gather_joint_modes(
                next_joint_token, mode_order
            )
            next_query_content = self._gather_query_modes(
                query_content, mode_order
            )
            next_agent_scores = self._gather_center_modes(
                pred_scores, mode_order
            )
            next_agent_trajs = self._gather_center_modes(
                refined_local, mode_order
            )
            assignment = self._gather_joint_modes(
                assignment, mode_order
            )
            next_intent_assignment = self._gather_joint_modes(
                next_intent_assignment, mode_order
            )

        state["joint_token"] = next_joint_token
        state["assignment"] = assignment
        state["intent_assignment"] = next_intent_assignment
        return {
            "query_content": next_query_content,
            "agent_pred_scores": next_agent_scores,
            "agent_pred_trajs": next_agent_trajs,
            "loss_agent_pred_scores": pred_scores,
            "loss_agent_pred_trajs": refined_local,
            "joint_logits": joint_logits,
            "base_joint_logits": base_joint_logits,
            "joint_trajs": refined_xy,
            "joint_anchor_trajs": canonical_xy,
            "assignment": assignment,
            "response": response,
            "traj_delta": applied_delta,
            "score_delta": applied_score,
            "query_gate": torch.sigmoid(self.query_gate_logits[layer_idx]),
            "feedback_gate": feedback_gate,
            "score_gate": score_gate,
            "traj_gate": traj_gate,
            "mode_order": mode_order,
            "horizon_reliability": reliability_ret,
        }

    def _selection_signature(self, trajs):
        batch_size, num_modes = trajs.shape[:2]
        if self.multihorizon_joint_nms:
            measurement_steps = torch.tensor(
                self.model_cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
                device=trajs.device,
                dtype=torch.long,
            ).clamp(0, trajs.shape[3] - 1)
            # [B, M, H, A*2]. RMS over horizons has the same scale as the
            # legacy final-endpoint distance when H=1.
            return trajs.index_select(3, measurement_steps).permute(
                0, 1, 3, 2, 4
            ).reshape(batch_size, num_modes, measurement_steps.numel(), -1)
        return trajs[..., -1, :].reshape(batch_size, num_modes, 1, 4)

    def _passes_selection_nms(self, signature, batch_idx, candidate, selected):
        return all(
            (
                (signature[batch_idx, candidate] - signature[batch_idx, kept])
                .square()
                .sum(dim=-1)
                .mean()
                .sqrt()
                .item()
            )
            >= self.nms_dist_thresh
            for kept in selected
        )

    def _selection_compatibility(self, trajs):
        """Pairwise form of the exact multi-horizon NMS distance test."""
        signature = self._selection_signature(trajs)
        distance = (
            (signature[:, :, None] - signature[:, None, :])
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
            .sqrt()
        )
        return distance >= self.nms_dist_thresh

    def _guarded_expansion_indices(
        self,
        final_output,
        *,
        max_replacements=None,
        admission_threshold=None,
        replacement_margin=None,
    ):
        """Select a protected expansion set under an explicit policy.

        The optional arguments are used by stacked candidate generators: the
        frozen legacy bank must keep the policy it was trained with even when a
        later augmentation stage owns the final deployment decision.
        """
        logits = final_output["joint_logits"]
        selection_logits = final_output.get(
            "protected_expansion_selection_logits", logits
        )
        trajs = final_output["joint_trajs"]
        admission_logits = final_output.get(
            "protected_expansion_admission_logits", None
        )
        if admission_logits is None:
            raise RuntimeError(
                "guarded_replace requires protected expansion admission logits"
            )
        batch_size, num_modes = logits.shape
        protected = self.protected_base_mode_count
        if num_modes != protected + admission_logits.shape[1]:
            raise RuntimeError("Expansion admission shape does not match modes")
        if selection_logits.shape != logits.shape:
            raise RuntimeError("Expansion selection logits do not match modes")

        configured_replacements = (
            self.max_expansion_replacements
            if max_replacements is None
            else int(max_replacements)
        )
        configured_admission = (
            self.expansion_admission_threshold
            if admission_threshold is None
            else float(admission_threshold)
        )
        configured_margin = (
            self.expansion_replacement_margin
            if replacement_margin is None
            else float(replacement_margin)
        )
        max_replacements = min(
            configured_replacements,
            self.num_output_modes,
            admission_logits.shape[1],
        )
        locked_count = self.num_output_modes - max_replacements
        signature = self._selection_signature(trajs)
        selected_rows = []
        for batch_idx in range(batch_size):
            base_order = selection_logits[batch_idx, :protected].argsort(
                descending=True
            ).tolist()
            locked = base_order[:locked_count]
            replaceable = base_order[locked_count:]
            selected = list(locked)
            expansion_count = 0
            base_floor = (
                selection_logits[batch_idx, replaceable].min()
                if replaceable
                else logits.new_tensor(float("inf"))
            )
            pool = replaceable + list(range(protected, num_modes))
            pool.sort(
                key=lambda index: float(
                    selection_logits[batch_idx, index].item()
                ),
                reverse=True,
            )
            for candidate in pool:
                is_expansion = candidate >= protected
                if is_expansion:
                    expansion_idx = candidate - protected
                    admission = torch.sigmoid(
                        admission_logits[batch_idx, expansion_idx]
                    )
                    if admission.item() < configured_admission:
                        continue
                    if expansion_count >= max_replacements:
                        continue
                    if (
                        selection_logits[batch_idx, candidate]
                        < base_floor + configured_margin
                    ):
                        continue
                if not self._passes_selection_nms(
                    signature, batch_idx, candidate, selected
                ):
                    continue
                selected.append(candidate)
                expansion_count += int(is_expansion)
                if len(selected) == self.num_output_modes:
                    break

            if len(selected) < self.num_output_modes:
                for candidate in base_order:
                    if candidate not in selected:
                        selected.append(candidate)
                    if len(selected) == self.num_output_modes:
                        break
            selected.sort(
                key=lambda index: float(logits[batch_idx, index].item()),
                reverse=True,
            )
            selected_rows.append(selected)
        return torch.tensor(
            selected_rows, device=logits.device, dtype=torch.long
        )

    @torch.no_grad()
    def _greedy_nms_indices(self, logits, trajs, num_output_modes=None):
        """Run the exact score-ordered NMS policy used for final prediction."""
        if num_output_modes is None:
            num_output_modes = self.num_output_modes
        num_output_modes = min(int(num_output_modes), logits.shape[1])
        compatibility = self._selection_compatibility(trajs)
        selected_rows = []
        for batch_idx in range(logits.shape[0]):
            order = logits[batch_idx].argsort(descending=True)
            selected = []
            for candidate_tensor in order:
                candidate = int(candidate_tensor.item())
                if not selected or bool(
                    compatibility[
                        batch_idx, candidate, selected
                    ].all().item()
                ):
                    selected.append(candidate)
                    if len(selected) == num_output_modes:
                        break
            if len(selected) < num_output_modes:
                for candidate_tensor in order:
                    candidate = int(candidate_tensor.item())
                    if candidate not in selected:
                        selected.append(candidate)
                        if len(selected) == num_output_modes:
                            break
            selected_rows.append(selected)
        return torch.tensor(
            selected_rows, device=logits.device, dtype=torch.long
        )

    def select_final(self, final_output):
        logits = final_output["joint_logits"]
        trajs = final_output["joint_trajs"]
        batch_size, num_modes = logits.shape
        if num_modes <= self.num_output_modes:
            return torch.softmax(logits, dim=-1), trajs

        if self.protected_expansion_enabled:
            if self.expansion_selection_mode == "base_only":
                base_logits = logits[:, : self.num_output_modes]
                base_trajs = trajs[:, : self.num_output_modes]
                return torch.softmax(base_logits, dim=-1), base_trajs
            selected = self._guarded_expansion_indices(final_output)
            batch_idx = torch.arange(
                batch_size, device=logits.device
            )[:, None]
            selected_logits = logits[batch_idx, selected]
            selected_trajs = trajs[batch_idx, selected]
            return torch.softmax(selected_logits, dim=-1), selected_trajs

        selected = self._greedy_nms_indices(logits, trajs)
        batch_idx = torch.arange(batch_size, device=logits.device)[:, None]
        selected_logits = logits[batch_idx, selected]
        selected_trajs = trajs[batch_idx, selected]
        return torch.softmax(selected_logits, dim=-1), selected_trajs

    def _official_match_quality(self, pred, input_dict):
        gt_state = input_dict["pair_gt_trajs"].to(pred.device).type_as(pred)
        gt = gt_state[..., 0:2]
        mask = input_dict["pair_gt_trajs_mask"].to(pred.device).bool()
        distance = torch.linalg.vector_norm(
            pred - gt[:, None], dim=-1
        )

        sample_idx = self.sample_indices.to(pred.device)
        sampled_dist = distance.index_select(3, sample_idx)
        sampled_valid = mask[:, None].index_select(3, sample_idx)
        ade = (sampled_dist * sampled_valid).sum(dim=(-1, -2)) / (
            sampled_valid.sum(dim=(-1, -2)).clamp_min(1)
        )

        horizon_idx = torch.tensor(
            self.model_cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
            device=pred.device,
            dtype=torch.long,
        ).clamp_max(self.num_future_frames - 1)
        horizon_error = (
            pred.index_select(3, horizon_idx)
            - gt[:, None].index_select(3, horizon_idx)
        )
        horizon_valid = mask.index_select(2, horizon_idx)
        horizon_distance = torch.linalg.vector_norm(
            horizon_error, dim=-1
        )
        fde = (horizon_distance * horizon_valid[:, None]).sum(
            dim=(-1, -2)
        ) / horizon_valid[:, None].sum(dim=(-1, -2)).clamp_min(1)

        gt_velocity = gt_state[..., 2:4].index_select(2, horizon_idx)
        gt_speed = torch.linalg.vector_norm(
            gt_velocity, dim=-1
        )
        pair_state = input_dict["pair_current_state"].to(
            pred.device
        ).type_as(pred)
        current_heading = torch.atan2(
            pair_state[..., 4], pair_state[..., 5]
        )
        if "pair_gt_trajs_src" in input_dict:
            gt_source = input_dict["pair_gt_trajs_src"].to(
                pred.device
            ).type_as(pred)
            history_samples = int(
                self.model_cfg.get("TRACK_HISTORY_SAMPLES", 10)
            )
            source_idx = (
                horizon_idx + history_samples + 1
            ).clamp_max(gt_source.shape[2] - 1)
            world_heading = gt_source[..., 6].index_select(
                2, source_idx
            )
            anchor_heading = input_dict[
                "pair_center_objects_world"
            ].to(pred.device).type_as(pred)[:, 0, 6]
            horizon_heading = (
                world_heading - anchor_heading[:, None, None]
            )
        else:
            horizon_heading = torch.atan2(
                gt_velocity[..., 1], gt_velocity[..., 0]
            )
            horizon_heading = torch.where(
                gt_speed > 0.5,
                horizon_heading,
                current_heading[..., None],
            )
        cos_heading = horizon_heading.cos()[:, None]
        sin_heading = horizon_heading.sin()[:, None]
        longitudinal = (
            horizon_error[..., 0] * cos_heading
            + horizon_error[..., 1] * sin_heading
        ).abs()
        lateral = (
            -horizon_error[..., 0] * sin_heading
            + horizon_error[..., 1] * cos_heading
        ).abs()

        initial_speed = torch.linalg.vector_norm(
            pair_state[..., 2:4], dim=-1
        )
        speed_scale = 0.5 + 0.5 * (
            (initial_speed - 1.4) / (11.0 - 1.4)
        ).clamp(0.0, 1.0)
        lateral_base = pred.new_tensor(
            self.model_cfg.get(
                "OFFICIAL_LATERAL_THRESHOLDS", [1.0, 1.8, 3.0]
            )
        )
        longitudinal_base = pred.new_tensor(
            self.model_cfg.get(
                "OFFICIAL_LONGITUDINAL_THRESHOLDS", [2.0, 3.6, 6.0]
            )
        )
        if lateral_base.numel() != horizon_idx.numel():
            raise ValueError(
                "OFFICIAL_LATERAL_THRESHOLDS must match MEASUREMENT_STEPS"
            )
        if longitudinal_base.numel() != horizon_idx.numel():
            raise ValueError(
                "OFFICIAL_LONGITUDINAL_THRESHOLDS must match "
                "MEASUREMENT_STEPS"
            )
        lateral_threshold = (
            speed_scale[:, None, :, None]
            * lateral_base[None, None, None]
        )
        longitudinal_threshold = (
            speed_scale[:, None, :, None]
            * longitudinal_base[None, None, None]
        )
        agent_match = (
            (lateral < lateral_threshold)
            & (longitudinal < longitudinal_threshold)
        )
        pair_valid = horizon_valid.all(dim=1)
        horizon_match = agent_match.all(dim=2) & pair_valid[:, None]
        joint_match = (
            horizon_match | ~pair_valid[:, None]
        ).all(dim=-1) & pair_valid.any(dim=-1, keepdim=True)

        normalized_error = torch.maximum(
            lateral / lateral_threshold.clamp_min(self.eps),
            longitudinal / longitudinal_threshold.clamp_min(self.eps),
        )
        normalized_error = normalized_error.masked_fill(
            ~horizon_valid[:, None], 0.0
        )
        horizon_cost = normalized_error.max(dim=2)[0]
        official_cost = (
            horizon_cost * pair_valid[:, None]
        ).sum(dim=-1) / pair_valid[:, None].sum(dim=-1).clamp_min(1)
        quality = (
            ade
            + float(self.model_cfg.get("FDE_WEIGHT", 0.35)) * fde
            + float(self.model_cfg.get("MISS_WEIGHT", 0.8))
            * official_cost
        )
        return {
            "gt": gt,
            "mask": mask,
            "quality": quality,
            "ade": ade,
            "fde": fde,
            "joint_match": joint_match,
            "horizon_match": horizon_match,
            "horizon_cost": horizon_cost,
            "pair_valid": pair_valid,
            "official_cost": official_cost,
        }

    def _select_ma_emta_positive(self, metrics):
        quality = metrics["quality"]
        joint_match = metrics["joint_match"]
        has_full_match = joint_match.any(dim=-1)
        earliest_full_match = joint_match.float().argmax(dim=-1)
        fallback = quality.argmin(dim=-1)

        horizon_match = metrics["horizon_match"]
        pair_valid = metrics["pair_valid"]
        matched_horizons = (
            horizon_match & pair_valid[:, None]
        ).sum(dim=-1)
        max_matched_horizons = matched_horizons.max(dim=-1)[0]

        if self.use_progressive_emta:
            best_coverage = (
                matched_horizons == max_matched_horizons[:, None]
            ) & (max_matched_horizons[:, None] > 0)
            earliest_best_coverage = best_coverage.float().argmax(dim=-1)
            fallback = torch.where(
                max_matched_horizons > 0,
                earliest_best_coverage,
                fallback,
            )

        positive_idx = torch.where(
            has_full_match, earliest_full_match, fallback
        )
        return positive_idx, matched_horizons, max_matched_horizons

    def _binary_focal_loss(self, logits, target):
        target = target.type_as(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = float(self.model_cfg.get("FOCAL_ALPHA", 0.75))
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        gamma = float(self.model_cfg.get("FOCAL_GAMMA", 2.0))
        return (
            alpha_t * (1.0 - p_t).pow(gamma) * ce
        ).mean(dim=-1)

    def _pair_sample_weights(self, input_dict, device, dtype):
        pair_types = input_dict.get("pair_object_types", None)
        type_weights = self.model_cfg.get(
            "PAIR_TYPE_LOSS_WEIGHTS", None
        )
        if pair_types is None or type_weights is None:
            return torch.ones(
                input_dict["pair_gt_trajs"].shape[0],
                device=device,
                dtype=dtype,
            )
        weights = []
        for pair in pair_types:
            names = {str(name) for name in pair}
            if "TYPE_CYCLIST" in names:
                pair_type = "TYPE_CYCLIST"
            elif "TYPE_PEDESTRIAN" in names:
                pair_type = "TYPE_PEDESTRIAN"
            else:
                pair_type = "TYPE_VEHICLE"
            weights.append(float(type_weights.get(pair_type, 1.0)))
        weights = torch.tensor(weights, device=device, dtype=dtype)
        return weights / weights.mean().clamp_min(self.eps)

    @staticmethod
    def _weighted_batch_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _ma_emta_prediction_loss(self, output, input_dict):
        pred = output["joint_trajs"]
        logits = output["joint_logits"]
        metrics = self._official_match_quality(pred, input_dict)
        quality = metrics["quality"]
        joint_match = metrics["joint_match"]
        (
            positive_idx,
            matched_horizons,
            max_matched_horizons,
        ) = self._select_ma_emta_positive(metrics)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        sample_weights = self._pair_sample_weights(
            input_dict, pred.device, pred.dtype
        )

        agent_pred = output["loss_agent_pred_trajs"]
        agent_scores = output["loss_agent_pred_scores"]
        center_gt = input_dict["center_gt_trajs"].to(
            pred.device
        ).type_as(pred)
        center_mask = input_dict["center_gt_trajs_mask"].to(
            pred.device
        ).type_as(pred)
        center_positive_idx = positive_idx[:, None].expand(
            -1, 2
        ).reshape(-1)
        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(
            pred_scores=agent_scores,
            pred_trajs=agent_pred[..., 0:5],
            gt_trajs=center_gt[..., 0:2],
            gt_valid_mask=center_mask,
            pre_nearest_mode_idxs=center_positive_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        valid_steps = center_mask.sum(dim=-1).clamp_min(1.0)
        loss_reg_gmm_sample = (
            loss_reg_gmm / valid_steps
        ).reshape(pred.shape[0], 2).mean(dim=-1)
        loss_reg_gmm = self._weighted_batch_mean(
            loss_reg_gmm_sample, sample_weights
        )
        selected_agent_pred = agent_pred[
            torch.arange(agent_pred.shape[0], device=pred.device),
            center_positive_idx,
        ]
        velocity_loss = F.smooth_l1_loss(
            selected_agent_pred[..., 5:7],
            center_gt[..., 2:4],
            reduction="none",
        ).sum(dim=-1)
        loss_velocity_sample = (
            (velocity_loss * center_mask).sum(dim=-1) / valid_steps
        ).reshape(pred.shape[0], 2).mean(dim=-1)
        loss_velocity = self._weighted_batch_mean(
            loss_velocity_sample, sample_weights
        )

        best = pred[batch_idx, positive_idx]
        reg = F.smooth_l1_loss(
            best, metrics["gt"], reduction="none"
        ).sum(dim=-1)
        loss_huber_sample = (
            (reg * metrics["mask"]).sum(dim=(-1, -2))
            / metrics["mask"].sum(dim=(-1, -2)).clamp_min(1)
        )
        loss_huber = self._weighted_batch_mean(
            loss_huber_sample, sample_weights
        )
        loss_reg = (
            loss_reg_gmm
            + float(self.model_cfg.get("LOSS_WEIGHT_VEL", 0.5))
            * loss_velocity
            + float(self.model_cfg.get("LOSS_WEIGHT_HUBER", 0.5))
            * loss_huber
        )

        score_target = torch.zeros_like(logits)
        score_target[batch_idx, positive_idx] = 1.0
        loss_score = self._weighted_batch_mean(
            self._binary_focal_loss(logits, score_target), sample_weights
        )
        best_logit = logits[batch_idx, positive_idx][:, None]
        non_best = torch.ones_like(logits, dtype=torch.bool)
        non_best[batch_idx, positive_idx] = False
        loss_rank_sample = F.softplus(
            logits[non_best].reshape(logits.shape[0], -1)
            - best_logit
            + float(self.model_cfg.get("RANK_MARGIN", 0.2))
        ).mean(dim=-1)
        loss_rank = self._weighted_batch_mean(
            loss_rank_sample, sample_weights
        )

        loss_listwise = logits.new_zeros(())
        if self.use_horizon_listwise:
            listwise_temperature = max(
                float(
                    self.model_cfg.get("LISTWISE_TEMPERATURE", 0.7)
                ),
                self.eps,
            )
            horizon_target = torch.softmax(
                -metrics["horizon_cost"].detach()
                / listwise_temperature,
                dim=1,
            )
            log_probability = F.log_softmax(logits, dim=-1)[..., None]
            horizon_ce = -(
                horizon_target * log_probability
            ).sum(dim=1)
            valid_horizon = metrics["pair_valid"].type_as(horizon_ce)
            loss_listwise_sample = (
                horizon_ce * valid_horizon
            ).sum(dim=-1) / valid_horizon.sum(dim=-1).clamp_min(1.0)
            loss_listwise = self._weighted_batch_mean(
                loss_listwise_sample, sample_weights
            )

        endpoints = pred[..., -1, :].reshape(pred.shape[0], pred.shape[1], 4)
        pairwise = torch.cdist(endpoints, endpoints)
        eye = torch.eye(
            pred.shape[1], device=pred.device, dtype=torch.bool
        )[None]
        loss_diversity_sample = F.relu(
            float(self.model_cfg.get("DIVERSITY_MARGIN", 3.0))
            - pairwise.masked_fill(eye, 1e6)
        ).mean(dim=(-1, -2))
        loss_diversity = self._weighted_batch_mean(
            loss_diversity_sample, sample_weights
        )
        pred_relative = best[:, 0] - best[:, 1]
        gt_relative = metrics["gt"][:, 0] - metrics["gt"][:, 1]
        relative_mask = metrics["mask"][:, 0] & metrics["mask"][:, 1]
        interaction = F.smooth_l1_loss(
            pred_relative, gt_relative, reduction="none"
        ).sum(dim=-1)
        loss_interaction_sample = (
            (interaction * relative_mask).sum(dim=-1)
            / relative_mask.sum(dim=-1).clamp_min(1)
        )
        loss_interaction = self._weighted_batch_mean(
            loss_interaction_sample, sample_weights
        )
        acceleration = (
            best[..., 2:, :]
            - 2.0 * best[..., 1:-1, :]
            + best[..., :-2, :]
        )
        loss_smooth = self._weighted_batch_mean(
            acceleration.abs().mean(dim=(-1, -2, -3)), sample_weights
        )

        total = (
            float(self.model_cfg.get("LOSS_WEIGHT_REG", 1.0)) * loss_reg
            + float(self.model_cfg.get("LOSS_WEIGHT_SCORE", 1.0)) * loss_score
            + float(self.model_cfg.get("LOSS_WEIGHT_RANK", 0.2)) * loss_rank
            + float(self.model_cfg.get("LOSS_WEIGHT_LISTWISE", 0.0))
            * loss_listwise
            + float(self.model_cfg.get("LOSS_WEIGHT_DIVERSITY", 0.05))
            * loss_diversity
            + float(self.model_cfg.get("LOSS_WEIGHT_INTERACTION", 0.2))
            * loss_interaction
            + float(self.model_cfg.get("LOSS_WEIGHT_SMOOTH", 0.01))
            * loss_smooth
        )
        horizon_match = metrics["horizon_match"].any(dim=1).float()
        valid_horizon = metrics["pair_valid"].float()
        horizon_coverage = (
            horizon_match * valid_horizon
        ).sum(dim=0) / valid_horizon.sum(dim=0).clamp_min(1.0)
        return total, {
            "loss_reg": loss_reg,
            "loss_reg_gmm": loss_reg_gmm,
            "loss_velocity": loss_velocity,
            "loss_huber": loss_huber,
            "loss_score": loss_score,
            "loss_rank": loss_rank,
            "loss_listwise": loss_listwise,
            "loss_diversity": loss_diversity,
            "loss_interaction": loss_interaction,
            "loss_smooth": loss_smooth,
            "oracle_ade": metrics["ade"].min(dim=-1)[0].mean(),
            "oracle_fde": metrics["fde"].min(dim=-1)[0].mean(),
            "oracle_miss": (~joint_match.any(dim=-1)).float().mean(),
            "positive_rank": positive_idx.float().mean(),
            "positive_horizon_count": matched_horizons[
                batch_idx, positive_idx
            ].float().mean(),
            "max_horizon_count": max_matched_horizons.float().mean(),
            "coverage_3s": horizon_coverage[0],
            "coverage_5s": horizon_coverage[1],
            "coverage_8s": horizon_coverage[2],
        }

    def _horizon_reliability_loss(self, output, input_dict):
        reliability = output.get("horizon_reliability", None)
        if reliability is None:
            raise RuntimeError(
                "Final decoder output is missing horizon reliability values"
            )
        pred = output["joint_trajs"].detach()
        metrics = self._official_match_quality(pred, input_dict)
        horizon_logits = reliability["horizon_logits"]
        soft_map_credit = build_soft_map_credit_targets(
            horizon_match=metrics["horizon_match"],
            horizon_cost=metrics["horizon_cost"].detach(),
            pair_valid=metrics["pair_valid"],
        )
        official_match_target = metrics["horizon_match"].type_as(
            horizon_logits
        )
        target = soft_map_credit["credited_target"].type_as(horizon_logits)
        supervision_mask = soft_map_credit["supervision_mask"].type_as(
            horizon_logits
        )
        if horizon_logits.shape != target.shape:
            raise RuntimeError(
                "Reliability output must match [batch, mode, horizon] targets"
            )
        sample_weights = self._pair_sample_weights(
            input_dict, pred.device, pred.dtype
        )

        ce = F.binary_cross_entropy_with_logits(
            horizon_logits, target, reduction="none"
        )
        probability = torch.sigmoid(horizon_logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = float(
            self.horizon_reliability_cfg.get("FOCAL_ALPHA", 0.75)
        )
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        gamma = float(
            self.horizon_reliability_cfg.get("FOCAL_GAMMA", 2.0)
        )
        focal = alpha_t * (1.0 - p_t).pow(gamma) * ce
        denominator = supervision_mask.sum(dim=(1, 2)).clamp_min(1.0)
        loss_calibration_sample = (focal * supervision_mask).sum(
            dim=(1, 2)
        ) / denominator
        loss_calibration = self._weighted_batch_mean(
            loss_calibration_sample, sample_weights
        )

        brier_sample = (
            (probability - target).square() * supervision_mask
        ).sum(dim=(1, 2)) / denominator
        loss_brier = self._weighted_batch_mean(
            brier_sample, sample_weights
        )

        has_match = soft_map_credit["has_match"]
        matched_distribution = target
        fallback_temperature = max(
            float(
                self.horizon_reliability_cfg.get(
                    "FALLBACK_TEMPERATURE", 0.7
                )
            ),
            self.eps,
        )
        fallback_distribution = torch.softmax(
            -metrics["horizon_cost"].detach() / fallback_temperature,
            dim=1,
        )
        target_distribution = torch.where(
            has_match[:, None],
            matched_distribution,
            fallback_distribution,
        )
        horizon_scores = (
            reliability["base_logits"].detach()[..., None]
            + horizon_logits
        )
        horizon_listwise = -(
            target_distribution
            * F.log_softmax(horizon_scores, dim=1)
        ).sum(dim=1)
        valid_horizon = metrics["pair_valid"].type_as(horizon_listwise)
        loss_horizon_listwise_sample = (
            horizon_listwise * valid_horizon
        ).sum(dim=-1) / valid_horizon.sum(dim=-1).clamp_min(1.0)
        loss_horizon_listwise = self._weighted_batch_mean(
            loss_horizon_listwise_sample, sample_weights
        )

        horizon_weights = self.world_reliability_head.horizon_weights.type_as(
            target_distribution
        )[None] * valid_horizon
        horizon_weights = horizon_weights / horizon_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        aggregate_target = (
            target_distribution * horizon_weights[:, None]
        ).sum(dim=-1)
        aggregate_target = aggregate_target / aggregate_target.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        fused_logits = reliability["fused_logits"]
        loss_fused_listwise_sample = -(
            aggregate_target * F.log_softmax(fused_logits, dim=-1)
        ).sum(dim=-1)
        loss_fused_listwise = self._weighted_batch_mean(
            loss_fused_listwise_sample, sample_weights
        )

        base_probability = F.softmax(
            reliability["base_logits"].detach(), dim=-1
        )
        trust_kl_sample = (
            base_probability
            * (
                torch.log(base_probability.clamp_min(self.eps))
                - F.log_softmax(fused_logits, dim=-1)
            )
        ).sum(dim=-1)
        loss_trust = self._weighted_batch_mean(
            trust_kl_sample, sample_weights
        ).clamp_min(0.0)
        loss_residual = reliability["logit_residual"].square().mean()

        total = (
            float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_CALIBRATION", 1.0
                )
            )
            * loss_calibration
            + float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_BRIER", 0.25
                )
            )
            * loss_brier
            + float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_HORIZON_LISTWISE", 0.5
                )
            )
            * loss_horizon_listwise
            + float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_FUSED_LISTWISE", 1.0
                )
            )
            * loss_fused_listwise
            + float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_TRUST", 0.2
                )
            )
            * loss_trust
            + float(
                self.horizon_reliability_cfg.get(
                    "LOSS_WEIGHT_RESIDUAL", 0.01
                )
            )
            * loss_residual
        )

        base_top = reliability["base_logits"].argmax(dim=-1)
        fused_top = fused_logits.argmax(dim=-1)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        base_top_match = official_match_target[batch_idx, base_top]
        fused_top_match = official_match_target[batch_idx, fused_top]
        valid_denominator = valid_horizon.sum().clamp_min(1.0)
        base_top_success = (
            base_top_match * valid_horizon
        ).sum() / valid_denominator
        fused_top_success = (
            fused_top_match * valid_horizon
        ).sum() / valid_denominator

        return total, {
            "loss_reliability_calibration": loss_calibration,
            "loss_reliability_brier": loss_brier,
            "loss_reliability_horizon_listwise": loss_horizon_listwise,
            "loss_reliability_fused_listwise": loss_fused_listwise,
            "loss_reliability_trust": loss_trust,
            "loss_reliability_residual": loss_residual,
            "reliability_gate": reliability["gate"],
            "reliability_residual_abs": reliability[
                "logit_residual"
            ].abs().mean(),
            "reliability_base_top_success": base_top_success,
            "reliability_fused_top_success": fused_top_success,
            "reliability_probability_positive": (
                (probability * target).sum()
                / target.sum().clamp_min(1.0)
            ),
            "reliability_probability_negative": (
                (
                    probability
                    * (1.0 - target)
                    * supervision_mask
                ).sum()
                / (
                    (1.0 - target) * supervision_mask
                ).sum().clamp_min(1.0)
            ),
            "reliability_ignored_match_fraction": (
                soft_map_credit["ignored_match"].float().sum()
                / metrics["horizon_match"].float().sum().clamp_min(1.0)
            ),
            "reliability_credited_match_rate": (
                has_match.float().sum()
                / metrics["pair_valid"].float().sum().clamp_min(1.0)
            ),
            "joint_oracle_ade": metrics["ade"].min(dim=-1)[0].mean(),
            "joint_oracle_fde": metrics["fde"].min(dim=-1)[0].mean(),
        }

    def _prediction_loss(self, output, input_dict):
        if self.use_ma_emta:
            return self._ma_emta_prediction_loss(output, input_dict)
        pred = output["joint_trajs"]
        logits = output["joint_logits"]
        gt = input_dict["pair_gt_trajs"].to(pred.device).type_as(pred)[
            ..., 0:2
        ]
        mask = input_dict["pair_gt_trajs_mask"].to(pred.device).type_as(pred)
        valid = mask[:, None]
        distance = torch.linalg.vector_norm(pred - gt[:, None], dim=-1)

        sample_idx = self.sample_indices.to(pred.device)
        sampled_dist = distance.index_select(3, sample_idx)
        sampled_valid = valid.index_select(3, sample_idx)
        ade = (sampled_dist * sampled_valid).sum(dim=(-1, -2)) / (
            sampled_valid.sum(dim=(-1, -2)).clamp_min(1.0)
        )

        horizon_idx = torch.tensor(
            self.model_cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
            device=pred.device,
            dtype=torch.long,
        ).clamp_max(self.num_future_frames - 1)
        horizon_dist = distance.index_select(3, horizon_idx)
        horizon_valid = valid.index_select(3, horizon_idx)
        fde = (horizon_dist * horizon_valid).sum(dim=(-1, -2)) / (
            horizon_valid.sum(dim=(-1, -2)).clamp_min(1.0)
        )
        thresholds = pred.new_tensor(
            self.model_cfg.get("MISS_THRESHOLDS", [2.0, 4.0, 6.0])
        ).view(1, 1, 1, -1)
        miss_agent = torch.sigmoid(
            (horizon_dist - thresholds)
            / max(float(self.model_cfg.get("MISS_TEMPERATURE", 1.0)), 1e-4)
        )
        miss_agent = (miss_agent * horizon_valid).sum(dim=-1) / (
            horizon_valid.sum(dim=-1).clamp_min(1.0)
        )
        miss = miss_agent.max(dim=-1)[0]
        quality = (
            ade
            + float(self.model_cfg.get("FDE_WEIGHT", 0.35)) * fde
            + float(self.model_cfg.get("MISS_WEIGHT", 0.8)) * miss
        )
        best_idx = quality.argmin(dim=-1)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        best = pred[batch_idx, best_idx]

        reg = F.smooth_l1_loss(best, gt, reduction="none").sum(dim=-1)
        loss_reg = ((reg * mask).sum(dim=(-1, -2)) / mask.sum(
            dim=(-1, -2)
        ).clamp_min(1.0)).mean()
        temperature = max(
            float(self.model_cfg.get("QUALITY_TEMPERATURE", 0.7)), 1e-4
        )
        soft_target = torch.softmax(-quality.detach() / temperature, dim=-1)
        loss_score = -(
            soft_target * F.log_softmax(logits, dim=-1)
        ).sum(dim=-1).mean()
        best_logit = logits[batch_idx, best_idx][:, None]
        non_best = torch.ones_like(logits, dtype=torch.bool)
        non_best[batch_idx, best_idx] = False
        loss_rank = F.softplus(
            logits[non_best].reshape(logits.shape[0], -1)
            - best_logit
            + float(self.model_cfg.get("RANK_MARGIN", 0.2))
        ).mean()

        endpoints = pred[..., -1, :].reshape(pred.shape[0], pred.shape[1], 4)
        pairwise = torch.cdist(endpoints, endpoints)
        eye = torch.eye(
            pred.shape[1], device=pred.device, dtype=torch.bool
        )[None]
        loss_diversity = F.relu(
            float(self.model_cfg.get("DIVERSITY_MARGIN", 3.0))
            - pairwise.masked_fill(eye, 1e6)
        ).mean()

        pred_relative = best[:, 0] - best[:, 1]
        gt_relative = gt[:, 0] - gt[:, 1]
        relative_mask = mask[:, 0] * mask[:, 1]
        interaction = F.smooth_l1_loss(
            pred_relative, gt_relative, reduction="none"
        ).sum(dim=-1)
        loss_interaction = (
            (interaction * relative_mask).sum(dim=-1)
            / relative_mask.sum(dim=-1).clamp_min(1.0)
        ).mean()
        acceleration = (
            best[..., 2:, :]
            - 2.0 * best[..., 1:-1, :]
            + best[..., :-2, :]
        )
        loss_smooth = acceleration.abs().mean()

        total = (
            float(self.model_cfg.get("LOSS_WEIGHT_REG", 1.0)) * loss_reg
            + float(self.model_cfg.get("LOSS_WEIGHT_SCORE", 1.0)) * loss_score
            + float(self.model_cfg.get("LOSS_WEIGHT_RANK", 0.2)) * loss_rank
            + float(self.model_cfg.get("LOSS_WEIGHT_DIVERSITY", 0.05))
            * loss_diversity
            + float(self.model_cfg.get("LOSS_WEIGHT_INTERACTION", 0.2))
            * loss_interaction
            + float(self.model_cfg.get("LOSS_WEIGHT_SMOOTH", 0.01))
            * loss_smooth
        )
        return total, {
            "loss_reg": loss_reg,
            "loss_score": loss_score,
            "loss_rank": loss_rank,
            "loss_diversity": loss_diversity,
            "loss_interaction": loss_interaction,
            "loss_smooth": loss_smooth,
            "oracle_ade": ade.min(dim=-1)[0].mean(),
            "oracle_fde": fde.min(dim=-1)[0].mean(),
            "oracle_miss": miss.min(dim=-1)[0].mean(),
        }

    def _world_loss(self, state, outputs, input_dict):
        prior = state["prior_rollout"]
        posterior = state["posterior_rollout"]
        target = state["sampled_gt"]
        mask = state["sampled_mask"]
        if posterior is None or target is None:
            zero = prior.new_zeros(())
            return zero, {
                "loss_world_prior": zero,
                "loss_world_posterior": zero,
                "loss_world_kl": zero,
                "loss_world_probability": zero,
                "loss_world_diversity": zero,
                "loss_assignment_balance": zero,
                "loss_assignment_entropy": zero,
                "loss_mode_world_assignment": zero,
                "loss_mode_world_consistency": zero,
                "loss_positive_world_alignment": zero,
                "loss_intent_balance": zero,
                "loss_intent_entropy": zero,
                "world_prior_best_error": zero,
            }

        sample_weights = self._pair_sample_weights(
            input_dict, prior.device, prior.dtype
        )
        valid = mask[:, None].type_as(prior)
        valid_count = valid.sum(dim=(-1, -2)).clamp_min(1.0)

        def rollout_error(rollout):
            pos = torch.linalg.vector_norm(
                rollout[..., 0:2] - target[:, None, ..., 0:2], dim=-1
            )
            vel = torch.linalg.vector_norm(
                rollout[..., 2:4] - target[:, None, ..., 2:4], dim=-1
            )
            return (
                ((pos + 0.2 * vel) * valid).sum(dim=(-1, -2))
                / valid_count
            )

        prior_error = rollout_error(prior)
        posterior_error = rollout_error(posterior)
        target_prob = torch.softmax(
            -posterior_error.detach()
            / max(
                float(self.model_cfg.get("WORLD_TARGET_TEMPERATURE", 0.7)),
                self.eps,
            ),
            dim=-1,
        )
        loss_prior = self._weighted_batch_mean(
            (target_prob * prior_error).sum(dim=-1), sample_weights
        )
        loss_posterior = self._weighted_batch_mean(
            (target_prob * posterior_error).sum(dim=-1), sample_weights
        )

        prior_mu = state["prior_mu"]
        prior_logvar = state["prior_logvar"]
        posterior_mu = state["posterior_mu"]
        posterior_logvar = state["posterior_logvar"]
        kl = 0.5 * (
            prior_logvar
            - posterior_logvar
            + (
                posterior_logvar.exp()
                + (posterior_mu - prior_mu).pow(2)
            )
            / prior_logvar.exp().clamp_min(self.eps)
            - 1.0
        ).sum(dim=-1)
        loss_kl = self._weighted_batch_mean(
            (target_prob * kl).sum(dim=-1), sample_weights
        )
        loss_probability = self._weighted_batch_mean(
            -(
                target_prob
                * torch.log_softmax(state["world_logits"], dim=-1)
            ).sum(dim=-1),
            sample_weights,
        )

        endpoints = prior[..., -1, 0:2].reshape(
            prior.shape[0], self.num_worlds, 4
        )
        world_distance = torch.cdist(endpoints, endpoints)
        eye = torch.eye(
            self.num_worlds, device=prior.device, dtype=torch.bool
        )[None]
        loss_world_diversity_sample = F.relu(
            float(self.model_cfg.get("WORLD_DIVERSITY_MARGIN", 2.0))
            - world_distance.masked_fill(eye, 1e6)
        ).mean(dim=(-1, -2))
        loss_world_diversity = self._weighted_batch_mean(
            loss_world_diversity_sample, sample_weights
        )

        assignment = outputs[-1]["assignment"]
        usage = assignment.mean(dim=1)
        uniform = usage.new_full(usage.shape, 1.0 / self.num_worlds)
        loss_balance = F.kl_div(
            usage.clamp_min(self.eps).log(), uniform, reduction="batchmean"
        )
        loss_entropy = -(
            assignment.clamp_min(self.eps)
            * assignment.clamp_min(self.eps).log()
        ).sum(dim=-1).mean()

        loss_mode_world_assignment = prior.new_zeros(())
        loss_mode_world_consistency = prior.new_zeros(())
        loss_positive_world_alignment = prior.new_zeros(())
        if self.use_world_mode_alignment:
            final_output = outputs[-1]
            sampled_mode_xy = final_output["joint_trajs"].index_select(
                3, self.sample_indices.to(prior.device)
            )
            world_xy = prior[..., 0:2]
            alignment_valid = mask[:, None, None].type_as(prior)
            alignment_valid_count = alignment_valid.sum(
                dim=(-1, -2)
            ).clamp_min(1.0)
            mode_world_distance = torch.linalg.vector_norm(
                sampled_mode_xy[:, :, None] - world_xy[:, None],
                dim=-1,
            )
            mode_world_cost = (
                mode_world_distance * alignment_valid
            ).sum(dim=(-1, -2)) / alignment_valid_count
            alignment_temperature = max(
                float(
                    self.model_cfg.get(
                        "WORLD_MODE_TEMPERATURE", 1.0
                    )
                ),
                self.eps,
            )
            assignment_target = torch.softmax(
                -mode_world_cost.detach() / alignment_temperature,
                dim=-1,
            )
            assignment_log = assignment.clamp_min(self.eps).log()
            assignment_ce_sample = -(
                assignment_target * assignment_log
            ).sum(dim=-1).mean(dim=-1)
            loss_mode_world_assignment = self._weighted_batch_mean(
                assignment_ce_sample, sample_weights
            )
            consistency_sample = (
                assignment.detach() * mode_world_cost
            ).sum(dim=-1).mean(dim=-1)
            loss_mode_world_consistency = self._weighted_batch_mean(
                consistency_sample, sample_weights
            )

            match_metrics = self._official_match_quality(
                final_output["joint_trajs"], input_dict
            )
            positive_idx, _, _ = self._select_ma_emta_positive(
                match_metrics
            )
            batch_idx = torch.arange(
                assignment.shape[0], device=assignment.device
            )
            positive_assignment = assignment[
                batch_idx, positive_idx
            ].clamp_min(self.eps)
            positive_alignment_sample = -(
                target_prob.detach() * positive_assignment.log()
            ).sum(dim=-1)
            loss_positive_world_alignment = self._weighted_batch_mean(
                positive_alignment_sample, sample_weights
            )

        intent_assignment = state["intent_assignment"]
        intent_usage = intent_assignment.mean(dim=1)
        intent_uniform = intent_usage.new_full(
            intent_usage.shape, 1.0 / intent_usage.shape[-1]
        )
        loss_intent_balance = F.kl_div(
            intent_usage.clamp_min(self.eps).log(),
            intent_uniform,
            reduction="batchmean",
        )
        loss_intent_entropy = -(
            intent_assignment.clamp_min(self.eps)
            * intent_assignment.clamp_min(self.eps).log()
        ).sum(dim=-1).mean()
        total = (
            float(self.model_cfg.get("LOSS_WEIGHT_WORLD_PRIOR", 0.15))
            * loss_prior
            + float(self.model_cfg.get("LOSS_WEIGHT_WORLD_POSTERIOR", 0.05))
            * loss_posterior
            + float(self.model_cfg.get("LOSS_WEIGHT_WORLD_KL", 0.005))
            * loss_kl
            + float(self.model_cfg.get("LOSS_WEIGHT_WORLD_PROB", 0.05))
            * loss_probability
            + float(self.model_cfg.get("LOSS_WEIGHT_WORLD_DIVERSITY", 0.02))
            * loss_world_diversity
            + float(self.model_cfg.get("LOSS_WEIGHT_ASSIGNMENT_BALANCE", 0.01))
            * loss_balance
            + float(self.model_cfg.get("LOSS_WEIGHT_ASSIGNMENT_ENTROPY", 0.001))
            * loss_entropy
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_MODE_WORLD_ASSIGNMENT", 0.0
                )
            )
            * loss_mode_world_assignment
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_MODE_WORLD_CONSISTENCY", 0.0
                )
            )
            * loss_mode_world_consistency
            + float(
                self.model_cfg.get(
                    "LOSS_WEIGHT_POSITIVE_WORLD_ALIGNMENT", 0.0
                )
            )
            * loss_positive_world_alignment
            + float(self.model_cfg.get("LOSS_WEIGHT_INTENT_BALANCE", 0.01))
            * loss_intent_balance
            + float(self.model_cfg.get("LOSS_WEIGHT_INTENT_ENTROPY", 0.001))
            * loss_intent_entropy
        )
        return total, {
            "loss_world_prior": loss_prior,
            "loss_world_posterior": loss_posterior,
            "loss_world_kl": loss_kl,
            "loss_world_probability": loss_probability,
            "loss_world_diversity": loss_world_diversity,
            "loss_assignment_balance": loss_balance,
            "loss_assignment_entropy": loss_entropy,
            "loss_mode_world_assignment": loss_mode_world_assignment,
            "loss_mode_world_consistency": loss_mode_world_consistency,
            "loss_positive_world_alignment": (
                loss_positive_world_alignment
            ),
            "loss_intent_balance": loss_intent_balance,
            "loss_intent_entropy": loss_intent_entropy,
            "world_prior_best_error": prior_error.min(dim=-1)[0].mean(),
        }

    def get_loss(self, state, outputs, input_dict):
        if self.train_stage == "horizon_reliability_warmup":
            reliability_loss, reliability_metrics = (
                self._horizon_reliability_loss(outputs[-1], input_dict)
            )
            tb = {
                "loss_integrated_joint_world": reliability_loss.item(),
                "loss_joint_prediction": 0.0,
                "loss_joint_world_dynamics": 0.0,
                "loss_joint_reg": 0.0,
                "loss_joint_score": 0.0,
                "loss_joint_rank": 0.0,
                "loss_joint_listwise": 0.0,
                "loss_joint_diversity": 0.0,
                "loss_joint_interaction": 0.0,
                "loss_joint_smooth": 0.0,
                "loss_joint_delta": 0.0,
                "loss_joint_delta_smooth": 0.0,
                "joint_query_gate": torch.sigmoid(
                    self.query_gate_logits
                ).mean().item(),
                "joint_feedback_gate": torch.sigmoid(
                    self.feedback_gate_logits
                ).mean().item(),
                "joint_score_gate": 1.0,
                "joint_traj_gate": torch.sigmoid(
                    self.traj_gate_logits
                ).mean().item(),
            }
            for key, value in reliability_metrics.items():
                tb[key] = (
                    value.item() if torch.is_tensor(value) else float(value)
                )
            return reliability_loss, tb

        layer_weights = torch.tensor(
            self.model_cfg.get(
                "LAYER_LOSS_WEIGHTS", [0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
            ),
            device=outputs[-1]["joint_logits"].device,
            dtype=outputs[-1]["joint_logits"].dtype,
        )
        if layer_weights.numel() != len(outputs):
            layer_weights = torch.linspace(
                0.25,
                1.0,
                len(outputs),
                device=layer_weights.device,
                dtype=layer_weights.dtype,
            )
        layer_weights = layer_weights / layer_weights.sum()
        prediction_loss = outputs[-1]["joint_logits"].new_zeros(())
        final_metrics = None
        for weight, output in zip(layer_weights, outputs):
            layer_loss, metrics = self._prediction_loss(output, input_dict)
            prediction_loss = prediction_loss + weight * layer_loss
            final_metrics = metrics

        world_loss, world_metrics = self._world_loss(
            state, outputs, input_dict
        )
        delta_reg = torch.stack(
            [output["traj_delta"].abs().mean() for output in outputs]
        ).mean()
        delta_smooth = torch.stack(
            [
                (
                    output["traj_delta"][..., 2:, :]
                    - 2.0 * output["traj_delta"][..., 1:-1, :]
                    + output["traj_delta"][..., :-2, :]
                ).abs().mean()
                for output in outputs
            ]
        ).mean()
        total = (
            prediction_loss
            + world_loss
            + float(self.model_cfg.get("LOSS_WEIGHT_DELTA", 0.01))
            * delta_reg
            + float(self.model_cfg.get("LOSS_WEIGHT_DELTA_SMOOTH", 0.02))
            * delta_smooth
        )
        tb = {
            "loss_integrated_joint_world": total.item(),
            "loss_joint_prediction": prediction_loss.item(),
            "loss_joint_world_dynamics": world_loss.item(),
            "loss_joint_reg": final_metrics["loss_reg"].item(),
            "loss_joint_score": final_metrics["loss_score"].item(),
            "loss_joint_rank": final_metrics["loss_rank"].item(),
            "loss_joint_listwise": final_metrics.get(
                "loss_listwise", prediction_loss.new_zeros(())
            ).item(),
            "loss_joint_diversity": final_metrics["loss_diversity"].item(),
            "loss_joint_interaction": final_metrics[
                "loss_interaction"
            ].item(),
            "loss_joint_smooth": final_metrics["loss_smooth"].item(),
            "loss_joint_delta": delta_reg.item(),
            "loss_joint_delta_smooth": delta_smooth.item(),
            "joint_oracle_ade": final_metrics["oracle_ade"].item(),
            "joint_oracle_fde": final_metrics["oracle_fde"].item(),
            "joint_oracle_miss_proxy": final_metrics["oracle_miss"].item(),
            "joint_query_gate": torch.sigmoid(self.query_gate_logits).mean().item(),
            "joint_feedback_gate": torch.sigmoid(
                self.feedback_gate_logits
            ).mean().item(),
            "joint_score_gate": (
                1.0
                if self.direct_joint_score
                else torch.sigmoid(self.score_gate_logits).mean().item()
            ),
            "joint_traj_gate": torch.sigmoid(self.traj_gate_logits).mean().item(),
        }
        for key in (
            "loss_reg_gmm",
            "loss_velocity",
            "loss_huber",
            "positive_rank",
            "positive_horizon_count",
            "max_horizon_count",
            "coverage_3s",
            "coverage_5s",
            "coverage_8s",
        ):
            if key in final_metrics:
                tb[f"joint_{key}"] = final_metrics[key].item()
        for key, value in world_metrics.items():
            tb[key] = value.item()
        return total, tb
