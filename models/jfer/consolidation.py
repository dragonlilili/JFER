"""Reliability-constrained reduction from twelve candidates to six modes."""

import torch


def consolidate_candidates(model, output):
    """Apply base protection, admission checks, endpoint NMS, and backfill."""
    logits = output["joint_logits"]
    trajectories = output["joint_trajs"]
    if logits.shape[1] <= model.num_output_modes:
        return torch.softmax(logits, dim=-1), trajectories

    selected = model._guarded_expansion_indices(output)
    batch_index = torch.arange(logits.shape[0], device=logits.device)[:, None]
    selected_logits = logits[batch_index, selected]
    selected_trajectories = trajectories[batch_index, selected]
    return torch.softmax(selected_logits, dim=-1), selected_trajectories
