"""Core components of Joint Future Exploration and Reasoning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import common as common_utils
from .joint_reasoning import _build_mlp

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
