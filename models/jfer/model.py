"""JFER decoder: structured bank, complementary expansion, refinement, and guard."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .joint_reasoning import JointReasoningCore
from .spatial_context import build_candidate_spatial_evidence
from .candidate_refinement import WorldConditionedCandidateResidualFlow
from .scoring import WorldConditionedMetricScorer, official_ap_group_ids, official_pair_type_ids
from .structured_bank import WorldConditionedJointCandidateBank
from .complementary_expansion import CandidateBankProtectedExpansion
from .consolidation import consolidate_candidates
from .horizon_scoring import build_soft_map_credit_targets


class JFERCore(JointReasoningCore):
    """Run the four-stage JFER prediction path."""

    def __init__(self, cfg, query_dim, map_dim, num_future_frames, num_decoder_layers):
        super().__init__(
            cfg=cfg,
            query_dim=query_dim,
            map_dim=map_dim,
            num_future_frames=num_future_frames,
            num_decoder_layers=num_decoder_layers,
        )
        bank_cfg = cfg.get("CANDIDATE_BANK", {})
        if not bank_cfg.get("ENABLED", True):
            raise ValueError("CANDIDATE_BANK must be enabled")
        if not self.direct_sparse_modes or self.num_query_modes != self.num_output_modes:
            raise ValueError("JFER requires direct six-slot decoding")
        self.candidate_bank = WorldConditionedJointCandidateBank(
            cfg=bank_cfg,
            query_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
            num_output_modes=self.num_output_modes,
        )

        metric_cfg = cfg.get("METRIC_ALIGNED_SCORER", {})
        self.metric_aligned_scorer_cfg = metric_cfg
        self.metric_scorer = WorldConditionedMetricScorer(
            cfg=metric_cfg,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            measurement_steps=cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
            dt=self.dt,
        ) if metric_cfg.get("ENABLED", False) else None

        expansion_cfg = cfg.get("PROTECTED_EXPANSION", {})
        self.use_candidate_expansion = bool(expansion_cfg.get("ENABLED", False))
        if not self.use_candidate_expansion or not self.expand_after_decoder:
            raise ValueError("JFER requires post-decoder complementary expansion")
        self.expansion_novelty_weight = float(expansion_cfg.get("BANK_NOVELTY_WEIGHT", 1.0))
        self.expansion_diversity_weight = float(expansion_cfg.get("BANK_DIVERSITY_WEIGHT", 0.5))
        self.expansion_distance_scale = float(expansion_cfg.get("BANK_DISTANCE_SCALE", 5.0))
        self.candidate_expansion = CandidateBankProtectedExpansion(
            cfg=expansion_cfg,
            hidden_dim=self.hidden_dim,
            num_expansion_modes=self.num_expansion_modes,
            measurement_steps=cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
            num_future_frames=self.num_future_frames,
        )

        refinement_cfg = cfg.get("CANDIDATE_RESIDUAL_FLOW", {})
        self.use_candidate_residual_flow = bool(refinement_cfg.get("ENABLED", False))
        self.candidate_residual_flow_cfg = refinement_cfg
        self.candidate_residual_flow = WorldConditionedCandidateResidualFlow(
            cfg=refinement_cfg,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            measurement_steps=cfg.get("MEASUREMENT_STEPS", [29, 49, 79]),
            num_future_frames=self.num_future_frames,
            dt=self.dt,
        ) if self.use_candidate_residual_flow else None

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

    def train_candidate_expansion_modules(self):
        if self.candidate_expansion is None:
            raise RuntimeError(
                "candidate_bank_expansion requires "
                "PROTECTED_EXPANSION.ENABLED=True"
            )
        self.training = False
        self.candidate_expansion.train(True)

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

    def initialize_queries(self, intention_query, intention_points, state):
        base_query, base_points, base_content, state = super().initialize_queries(
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
        base_query = base_query.permute(1, 0, 2).reshape(
            batch_size, 2, self.num_output_modes, self.query_dim
        ).permute(0, 2, 1, 3)
        base_points = base_points.permute(1, 0, 2).reshape(
            batch_size, 2, self.num_output_modes, 2
        ).permute(0, 2, 1, 3)
        base_content = base_content.permute(1, 0, 2).reshape(
            batch_size, 2, self.num_output_modes, self.query_dim
        ).permute(0, 2, 1, 3)
        bank = self.candidate_bank(
            anchor_hidden=anchor_hidden,
            anchor_query=anchor_query,
            anchor_points=anchor_points,
            base_query=base_query,
            base_points=base_points,
            base_query_content=base_content,
            base_joint_token=state["joint_token"],
            base_intent_assignment=state["intent_assignment"],
            state=state,
        )
        adapted_query = bank["adapted_query"].permute(0, 2, 1, 3).reshape(
            batch_size * 2, self.num_output_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        adapted_points = bank["adapted_points"].permute(0, 2, 1, 3).reshape(
            batch_size * 2, self.num_output_modes, 2
        ).permute(1, 0, 2).contiguous()
        adapted_content = bank["adapted_query_content"].permute(0, 2, 1, 3).reshape(
            batch_size * 2, self.num_output_modes, self.query_dim
        ).permute(1, 0, 2).contiguous()
        state["joint_token"] = bank["adapted_joint_token"]
        state["intent_assignment"] = bank["adapted_intent_assignment"]
        state["assignment"], _, _ = self._assign_worlds(state["joint_token"], state)
        state["candidate_bank"] = bank
        state["candidate_bank_log_prior"] = bank["slot_pair_log_prior"]
        return adapted_query, adapted_points, adapted_content, state


    def _select_complementary_pair_indices(self, bank, base_trajs, base_hidden, base_logits):
        pair_points = bank["pair_points_canonical"].detach()
        pair_logits = bank["pair_logits"].detach()
        batch_size, num_pairs = pair_logits.shape
        base_endpoint = base_trajs[..., -1, :].reshape(
            batch_size, self.num_output_modes, -1
        ).detach()
        pair_endpoint = pair_points.reshape(batch_size, num_pairs, -1)
        novelty = torch.cdist(pair_endpoint.float(), base_endpoint.float()).amin(
            dim=-1
        ).type_as(pair_logits)
        normalized_prior = pair_logits - pair_logits.mean(dim=-1, keepdim=True)
        normalized_prior = normalized_prior / pair_logits.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        proposal = self.candidate_expansion.score_pair_bank(
            pair_hidden=bank["pair_hidden"].detach(),
            pair_endpoints=pair_points,
            pair_logits=pair_logits,
            base_hidden=base_hidden.detach(),
            base_logits=base_logits.detach(),
            base_trajs=base_trajs.detach(),
        )
        proposal["selection_logits"] = (
            self.candidate_expansion.pair_selector_prior_weight * normalized_prior
            + self.candidate_expansion.pair_selector_novelty_weight
            * self.expansion_novelty_weight
            * torch.tanh(novelty / max(self.expansion_distance_scale, self.eps))
            + proposal["selection_delta"]
        )
        used = torch.zeros_like(pair_logits, dtype=torch.bool)
        if not self.candidate_expansion.allow_base_pair_reuse:
            used.scatter_(1, bank["hard_pair_indices"].detach(), True)
        selected = []
        for _ in range(self.num_expansion_modes):
            score = proposal["selection_logits"]
            if selected:
                indices = torch.stack(selected, dim=1)
                endpoints = pair_endpoint.gather(
                    1, indices[..., None].expand(-1, -1, pair_endpoint.shape[-1])
                )
                diversity = torch.cdist(
                    pair_endpoint.float(), endpoints.float()
                ).amin(dim=-1).type_as(pair_logits)
                score = score + self.expansion_diversity_weight * torch.tanh(
                    diversity / max(self.expansion_distance_scale, self.eps)
                )
            index = score.masked_fill(used, -torch.inf).argmax(dim=-1)
            selected.append(index)
            used.scatter_(1, index[:, None], True)
        return torch.stack(selected, dim=1), proposal


    def _append_candidate_expansions(self, output, state):
        bank = state["candidate_bank"]
        base_trajs = output["joint_trajs"].detach()
        base_logits = output["joint_logits"].detach()
        base_hidden = output["response"].detach()
        pair_indices, proposal = self._select_complementary_pair_indices(
            bank, base_trajs, base_hidden, base_logits
        )
        pair_hidden = WorldConditionedJointCandidateBank._gather_modes(
            bank["pair_hidden"], pair_indices
        ).detach()
        pair_endpoints = WorldConditionedJointCandidateBank._gather_modes(
            bank["pair_points_canonical"], pair_indices
        ).detach()
        donor_indices = proposal["donor_indices"].gather(1, pair_indices)
        donor_hidden = base_hidden.gather(
            1, donor_indices[..., None].expand(-1, -1, base_hidden.shape[-1])
        )
        donor_logits = base_logits.gather(1, donor_indices)
        normalized_prior = bank["pair_logits"].detach()
        normalized_prior = normalized_prior - normalized_prior.mean(dim=-1, keepdim=True)
        normalized_prior = normalized_prior / normalized_prior.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(0.25)
        proposal_prior = normalized_prior.gather(1, pair_indices)
        donor_trajs = base_trajs.gather(
            1, donor_indices[..., None, None, None].expand(
                -1, -1, base_trajs.shape[2], base_trajs.shape[3], base_trajs.shape[4]
            )
        )
        proposal_trajs = None
        if self.candidate_expansion.use_pair_prototype_as_donor:
            proposal_trajs = WorldConditionedJointCandidateBank._gather_modes(
                proposal["prototype_trajs"], pair_indices
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
        expanded["protected_expansion_admission_logits"] = expansion["admission_logits"]
        expanded["protected_expansion_candidate_selector_logits"] = expansion["candidate_selector_logits"]
        expanded["protected_expansion_scene_gate_logits"] = expansion["scene_replacement_gate_logits"]
        expanded["protected_expansion_replacement_utility"] = expansion["replacement_utility"]
        expanded["candidate_expansion_horizon_logits"] = expansion["horizon_logits"]
        expanded["candidate_expansion_score_delta"] = expansion["score_delta"]
        expanded["candidate_expansion_waypoint_delta"] = expansion["waypoint_delta"]
        expanded["candidate_expansion_dense_knot_delta"] = expansion["dense_knot_delta"]
        expanded["candidate_expansion_dense_temporal_delta"] = expansion["dense_temporal_delta"]
        expanded["candidate_expansion_goal_gate"] = expansion["goal_gate"]
        expanded["candidate_expansion_confidence_gate"] = expansion["confidence_gate"]
        expanded["candidate_expansion_unified_confidence_residual"] = expansion["unified_confidence_residual"]
        expanded["candidate_expansion_unified_confidence_temperature"] = expansion["unified_confidence_temperature"]
        base_world = output["response"][:, :, None].expand(
            -1, -1, expansion["horizon_token"].shape[2], -1
        ).detach()
        expanded["protected_expansion_world_hidden"] = torch.cat(
            [base_world, expansion["horizon_token"].detach()], dim=1
        )
        expanded["candidate_expansion_pair_indices"] = pair_indices
        expanded["candidate_expansion_donor_indices"] = donor_indices
        expanded["candidate_pair_proposal_horizon_logits"] = proposal["horizon_logits"]
        expanded["candidate_pair_proposal_rescue_logits"] = proposal["rescue_logits"]
        expanded["candidate_pair_proposal_quality_logits"] = proposal["quality_logits"]
        expanded["candidate_pair_proposal_selection_delta"] = proposal["selection_delta"]
        expanded["candidate_pair_proposal_selection_logits"] = proposal["selection_logits"]
        expanded["candidate_pair_proposal_gate"] = proposal["gate"]
        expanded["candidate_pair_prototype_trajs"] = proposal["prototype_trajs"]
        expanded["candidate_pair_proposal_temporal_delta"] = proposal["temporal_waypoint_delta"]
        used_mask = torch.zeros_like(bank["pair_logits"], dtype=torch.bool)
        if not self.candidate_expansion.allow_base_pair_reuse:
            used_mask.scatter_(1, bank["hard_pair_indices"].detach(), True)
        expanded["candidate_pair_base_used_mask"] = used_mask
        return expanded


    def _apply_candidate_residual_flow(self, output, state, input_dict):
        module = self.candidate_residual_flow
        world_hidden = output.get("protected_expansion_world_hidden")
        if world_hidden is None:
            raise RuntimeError("Candidate refinement requires expansion world features")
        trajectories = output["joint_trajs"]
        logits = output["joint_logits"]
        selection_logits = output["protected_expansion_selection_logits"]
        spatial_evidence = build_candidate_spatial_evidence(
            canonical_trajectories=trajectories, input_dict=input_dict,
            measurement_steps=module.measurement_steps, dt=self.dt,
        ).type_as(world_hidden)
        detach_inputs = self.train_stage == "candidate_bank_residual_flow_geometry"
        if detach_inputs:
            trajectories = trajectories.detach()
            logits = logits.detach()
            selection_logits = selection_logits.detach()
            world_hidden = world_hidden.detach()
            spatial_evidence = spatial_evidence.detach()
            scene_context = state["scene_token"].detach()
            pair_state = state["pair_state"].detach()
        else:
            scene_context = state["scene_token"]
            pair_state = state["pair_state"]
        branch_ids = torch.arange(logits.shape[1], device=logits.device)[None].expand(
            logits.shape[0], -1
        ) >= self.num_output_modes
        flow_output = module(
            world_hidden=world_hidden, trajectories=trajectories, base_logits=logits,
            selection_logits=selection_logits, branch_ids=branch_ids,
            pair_type_ids=self._pair_type_ids(input_dict, logits.device, logits.shape[0]),
            scene_context=scene_context, spatial_evidence=spatial_evidence,
            pair_state=pair_state, enable_scoring=False,
        )
        refined = dict(output)
        refined["joint_trajs"] = flow_output["trajectories"]
        refined["joint_logits"] = flow_output["joint_logits"]
        refined["protected_expansion_selection_logits"] = flow_output["selection_logits"]
        admission = output.get("protected_expansion_admission_logits")
        if admission is not None:
            refined["protected_expansion_admission_logits"] = (
                admission + flow_output["admission_delta"][:, -admission.shape[1]:]
            )
        refined["candidate_residual_flow"] = flow_output
        refined["candidate_residual_flow_spatial_evidence"] = spatial_evidence
        return refined

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

    def _candidate_expansion_loss(
        self, final_output, input_dict, score_only=False
    ):
        expansion_module = self.candidate_expansion
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
        deployed_rate = (
            selected_expansion_mask.type_as(pred).mean()
            if selected_expansion_mask is not None
            else pred.new_zeros(())
        )
        return total, {
            "loss_candidate_expansion": total,
            "loss_expansion_coverage": loss_coverage,
            "loss_expansion_regression": loss_regression,
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

    def _base_generation_outputs(self, outputs):
        """Return decoder outputs before post-decoder candidate expansion.

        Protected expansion changes the final candidate axis from six base
        generation modes to a larger deployment pool. The base prediction
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

    def prepare_worlds(self, center_feature, obj_feature, obj_mask, map_feature, map_mask, input_dict, obj_pos=None, map_pos=None):
        state = super().prepare_worlds(
            center_feature=center_feature, obj_feature=obj_feature, obj_mask=obj_mask,
            map_feature=map_feature, map_mask=map_mask, input_dict=input_dict,
            obj_pos=obj_pos, map_pos=map_pos,
        )
        state["pair_center_world"] = input_dict["pair_center_objects_world"].to(
            center_feature.device
        ).type_as(center_feature)
        return state

    def condition_predictions(self, layer_idx, query_content, pred_scores, pred_trajs, state, input_dict):
        batch_size = state["batch_size"]
        num_modes = pred_scores.shape[1]
        marginal_log_prior = torch.log_softmax(
            pred_scores.reshape(batch_size, 2, num_modes), dim=-1
        ).sum(dim=1)
        output = super().condition_predictions(
            layer_idx=layer_idx, query_content=query_content, pred_scores=pred_scores,
            pred_trajs=pred_trajs, state=state, input_dict=input_dict,
        )
        if layer_idx + 1 < self.num_decoder_layers:
            state["candidate_bank_log_prior"] = state["candidate_bank_log_prior"].gather(
                1, output["mode_order"]
            )
            return output

        if self.metric_scorer is not None:
            spatial_evidence = build_candidate_spatial_evidence(
                canonical_trajectories=output["joint_trajs"],
                input_dict=input_dict,
                measurement_steps=self.metric_scorer.measurement_steps,
                dt=self.dt,
            ) if self.metric_scorer.use_spatial_evidence else None
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
            if self.train_stage == "candidate_bank_ap_calibration":
                scorer_inputs = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in scorer_inputs.items()
                }
            scorer_output = self.metric_scorer(
                **scorer_inputs,
                pair_type_ids=self._pair_type_ids(input_dict, output["joint_logits"].device, batch_size),
            )
            output["joint_pre_metric_logits"] = output["joint_logits"]
            output["joint_logits"] = scorer_output["joint_logits"]
            output["metric_aligned_scorer"] = scorer_output
            if spatial_evidence is not None:
                output["candidate_spatial_evidence"] = spatial_evidence

        output = self._append_candidate_expansions(output, state)
        if self.candidate_residual_flow is not None:
            output = self._apply_candidate_residual_flow(output, state, input_dict)
        return output

    def select_final(self, final_output):
        return consolidate_candidates(self, final_output)

    @staticmethod
    def _metrics_to_scalars(metrics):
        return {
            key: value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
            for key, value in metrics.items()
        }

    def get_loss(self, state, outputs, input_dict):
        final_output = outputs[-1]
        if self.train_stage == "candidate_bank_residual_flow_geometry":
            flow_output = final_output.get("candidate_residual_flow")
            if flow_output is None:
                raise RuntimeError("Missing candidate refinement output")
            refined_metrics = self._official_match_quality(flow_output["trajectories"], input_dict)
            with torch.no_grad():
                base_metrics = self._official_match_quality(flow_output["base_trajectories"], input_dict)
            loss, metrics = self.candidate_residual_flow.get_loss(
                output=flow_output, input_dict=input_dict,
                refined_metrics=refined_metrics, base_metrics=base_metrics, loss_mode="geometry",
            )
            tb = self._metrics_to_scalars(metrics)
            tb["loss_integrated_joint_world"] = loss.item()
            tb["joint_oracle_ade"] = refined_metrics["ade"].amin(dim=1).mean().item()
            tb["joint_oracle_fde"] = refined_metrics["fde"].amin(dim=1).mean().item()
            return loss, tb

        if self.train_stage == "candidate_bank_ap_calibration":
            metric_output = dict(final_output)
            mode_count = metric_output["metric_aligned_scorer"]["horizon_logits"].shape[1]
            metric_output["joint_trajs"] = metric_output["joint_trajs"][:, :mode_count]
            metric_output["joint_logits"] = metric_output["joint_logits"][:, :mode_count]
            loss, metrics = self._metric_aligned_scorer_loss(metric_output, input_dict)
            tb = self._metrics_to_scalars(metrics)
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb

        if self.train_stage in {"candidate_bank_expansion", "candidate_bank_expansion_calibration"}:
            loss, metrics = self._candidate_expansion_loss(final_output, input_dict)
            tb = self._metrics_to_scalars(metrics)
            tb["loss_integrated_joint_world"] = loss.item()
            return loss, tb

        base_loss, tb = super().get_loss(state, outputs, input_dict)
        bank_loss, bank_metrics = self.candidate_bank.get_loss(state["candidate_bank"], input_dict)
        total = base_loss + bank_loss
        tb.update(self._metrics_to_scalars(bank_metrics))
        tb["loss_integrated_joint_world"] = total.item()
        return total, tb
