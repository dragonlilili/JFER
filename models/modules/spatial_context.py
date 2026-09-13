"""Sparse spatial evidence for ranking joint trajectory candidates.

The candidate-bank decoder already produces useful trajectories.  This module
keeps those trajectories fixed and measures how each candidate interacts with
the scene at the official 3/5/8-second horizons.  All features are computed
from information available at inference time.
"""

import torch


SPATIAL_EVIDENCE_DIM = 22


def _rotate_xy(points, angle):
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    view_shape = [points.shape[0]] + [1] * (points.ndim - 2)
    cosine = cosine.view(*view_shape)
    sine = sine.view(*view_shape)
    x_coord = points[..., 0]
    y_coord = points[..., 1]
    return torch.stack(
        [
            x_coord * cosine - y_coord * sine,
            x_coord * sine + y_coord * cosine,
        ],
        dim=-1,
    )


def _masked_min(distance, mask, fallback):
    masked = distance.masked_fill(
        ~mask[:, None], torch.finfo(distance.dtype).max
    )
    value, index = masked.min(dim=-1)
    has_value = mask.any(dim=-1)[:, None]
    value = torch.where(
        has_value, value, value.new_full(value.shape, fallback)
    )
    index = torch.where(has_value, index, torch.zeros_like(index))
    return value, index


def _pair_local_positions(canonical_position, pair_center_world):
    """Convert anchor-0 coordinates to each target agent's local frame."""
    agent0 = pair_center_world[:, 0]
    agent1 = pair_center_world[:, 1]
    agent0_position = canonical_position[..., 0, :]

    anchor_shape = canonical_position[..., 1, :].shape
    anchor1 = canonical_position[..., 1, :].reshape(
        canonical_position.shape[0], -1, 2
    )
    world1 = _rotate_xy(anchor1, agent0[:, 6])
    world1 = world1 + agent0[:, None, 0:2]
    local1 = _rotate_xy(
        world1 - agent1[:, None, 0:2], -agent1[:, 6]
    ).reshape(anchor_shape)
    return torch.stack([agent0_position, local1], dim=-2)


def _pair_local_vectors(canonical_vector, pair_center_world):
    """Rotate anchor-0 vectors into each target agent's local frame."""
    agent0_vector = canonical_vector[..., 0, :]
    relative_heading = (
        pair_center_world[:, 0, 6] - pair_center_world[:, 1, 6]
    )
    vector_shape = canonical_vector[..., 1, :].shape
    agent1_vector = _rotate_xy(
        canonical_vector[..., 1, :].reshape(
            canonical_vector.shape[0], -1, 2
        ),
        relative_heading,
    ).reshape(vector_shape)
    return torch.stack([agent0_vector, agent1_vector], dim=-2)


def _reshape_pair_view(tensor, batch_size):
    if tensor.shape[0] != batch_size * 2:
        raise ValueError(
            "Expected a two-view interaction batch, got "
            f"{tensor.shape[0]} rows for {batch_size} scenes"
        )
    return tensor.reshape(batch_size, 2, *tensor.shape[1:])


def build_candidate_spatial_evidence(
    canonical_trajectories,
    input_dict,
    measurement_steps,
    dt=0.1,
):
    """Build candidate x horizon scene-consistency evidence.

    Args:
        canonical_trajectories: [B, M, 2, T, 2] in target-0 coordinates.
        input_dict: collated Waymo interaction input dictionary.
        measurement_steps: zero-based metric horizon indices.
    Returns:
        Tensor [B, M, H, 22].
    """
    if canonical_trajectories.ndim != 5:
        raise ValueError("Expected trajectories with shape [B, M, 2, T, 2]")
    if canonical_trajectories.shape[2] != 2:
        raise ValueError("Spatial evidence requires a two-agent joint future")

    trajectories = canonical_trajectories.float()
    device = trajectories.device
    batch_size, num_modes = trajectories.shape[:2]
    steps = torch.as_tensor(
        measurement_steps, device=device, dtype=torch.long
    ).clamp(0, trajectories.shape[3] - 1)
    previous_steps = (steps - 5).clamp_min(0)
    position = trajectories.index_select(3, steps).permute(0, 1, 3, 2, 4)
    previous = trajectories.index_select(3, previous_steps).permute(
        0, 1, 3, 2, 4
    )
    elapsed = (
        (steps - previous_steps).clamp_min(1).float() * float(dt)
    )
    velocity = (position - previous) / elapsed[None, None, :, None, None]

    pair_center_world = input_dict["pair_center_objects_world"].to(
        device=device, dtype=trajectories.dtype
    )
    local_position = _pair_local_positions(position, pair_center_world)
    local_velocity = _pair_local_vectors(velocity, pair_center_world)
    num_horizons = steps.numel()
    pair_queries = local_position.permute(0, 3, 1, 2, 4).reshape(
        batch_size * 2, num_modes * num_horizons, 2
    )
    pair_velocity = local_velocity.permute(0, 3, 1, 2, 4).reshape(
        batch_size * 2, num_modes, num_horizons, 2
    )

    map_data = _reshape_pair_view(
        input_dict["map_polylines"].to(
            device=device, dtype=trajectories.dtype
        ),
        batch_size,
    ).reshape(batch_size * 2, -1, 9)
    map_mask = _reshape_pair_view(
        input_dict["map_polylines_mask"].to(device=device).bool(),
        batch_size,
    ).reshape(batch_size * 2, -1)
    map_mask &= torch.isfinite(map_data[..., 0:2]).all(dim=-1)
    map_type = map_data[..., 6].round().long()
    map_distance = torch.cdist(pair_queries, map_data[..., 0:2])

    lane_mask = map_mask & (
        (map_type == 1) | (map_type == 2) | (map_type == 3)
    )
    boundary_mask = map_mask & (
        (map_type == 15) | (map_type == 16)
    )
    crosswalk_mask = map_mask & (map_type == 18)
    lane_distance, lane_index = _masked_min(
        map_distance, lane_mask, fallback=50.0
    )
    boundary_distance, _ = _masked_min(
        map_distance, boundary_mask, fallback=50.0
    )
    crosswalk_distance, _ = _masked_min(
        map_distance, crosswalk_mask, fallback=50.0
    )
    lane_direction = map_data[..., 3:5].gather(
        1, lane_index[..., None].expand(-1, -1, 2)
    )
    lane_direction = lane_direction / torch.linalg.vector_norm(
        lane_direction, dim=-1, keepdim=True
    ).clamp_min(1e-3)
    flat_velocity = pair_velocity.reshape(
        batch_size * 2, num_modes * num_horizons, 2
    )
    speed = torch.linalg.vector_norm(
        flat_velocity, dim=-1, keepdim=True
    )
    direction = flat_velocity / speed.clamp_min(0.5)
    lane_alignment = (direction * lane_direction).sum(dim=-1)
    lane_alignment = torch.where(
        speed.squeeze(-1) > 0.5,
        lane_alignment,
        torch.zeros_like(lane_alignment),
    )
    map_features = torch.stack(
        [
            (lane_distance / 15.0).clamp(max=4.0),
            lane_alignment.clamp(-1.0, 1.0),
            (boundary_distance / 20.0).clamp(max=4.0),
            (crosswalk_distance / 25.0).clamp(max=4.0),
        ],
        dim=-1,
    ).reshape(batch_size, 2, num_modes, num_horizons, 4)
    map_features = map_features.permute(0, 2, 3, 1, 4).flatten(
        start_dim=-2
    )

    object_data = input_dict["obj_trajs"].to(
        device=device, dtype=trajectories.dtype
    )
    object_mask = input_dict["obj_trajs_mask"].to(device=device).bool()
    current_position = object_data[:, :, -1, 0:2]
    current_velocity = object_data[:, :, -1, 25:27]
    current_valid = object_mask[:, :, -1].clone()
    center_index = input_dict["track_index_to_predict"].to(
        device=device
    ).long()
    partner_index = input_dict["pair_partner_index"].to(
        device=device
    ).long().reshape(-1)
    row_index = torch.arange(batch_size * 2, device=device)
    current_valid[row_index, center_index] = False
    current_valid[row_index, partner_index] = False

    horizon_time = (steps.float() + 1.0) * float(dt)
    object_future = (
        current_position[:, None]
        + horizon_time[None, :, None, None] * current_velocity[:, None]
    )
    neighbor_delta = (
        pair_queries.reshape(
            batch_size * 2, num_modes, num_horizons, 2
        )[:, :, :, None]
        - object_future[:, None]
    )
    neighbor_distance = torch.linalg.vector_norm(
        neighbor_delta, dim=-1
    ).masked_fill(
        ~current_valid[:, None, None], torch.finfo(trajectories.dtype).max
    )
    nearest_distance, nearest_index = neighbor_distance.min(dim=-1)
    has_neighbor = current_valid.any(dim=-1)[:, None, None]
    nearest_distance = torch.where(
        has_neighbor,
        nearest_distance,
        nearest_distance.new_full(nearest_distance.shape, 50.0),
    )
    nearest_velocity = current_velocity.gather(
        1,
        nearest_index.reshape(batch_size * 2, -1)[..., None].expand(
            -1, -1, 2
        ),
    ).reshape(batch_size * 2, num_modes, num_horizons, 2)
    nearest_delta = neighbor_delta.gather(
        3, nearest_index[..., None, None].expand(-1, -1, -1, 1, 2)
    ).squeeze(3)
    relative_velocity = nearest_velocity[:, :, :, :] - pair_velocity
    neighbor_closing_speed = -(
        nearest_delta * relative_velocity
    ).sum(dim=-1) / nearest_distance.clamp_min(0.5)
    neighbor_features = torch.stack(
        [
            (nearest_distance / 30.0).clamp(max=3.0),
            (neighbor_closing_speed / 15.0).clamp(-2.0, 2.0),
            torch.exp(-nearest_distance / 4.0),
        ],
        dim=-1,
    ).reshape(batch_size, 2, num_modes, num_horizons, 3)
    neighbor_features = neighbor_features.permute(
        0, 2, 3, 1, 4
    ).flatten(start_dim=-2)

    pair_delta = position[..., 1, :] - position[..., 0, :]
    pair_relative_velocity = velocity[..., 1, :] - velocity[..., 0, :]
    pair_distance = torch.linalg.vector_norm(pair_delta, dim=-1)
    pair_closing_speed = -(
        pair_delta * pair_relative_velocity
    ).sum(dim=-1) / pair_distance.clamp_min(0.5)
    pair_speed = torch.linalg.vector_norm(
        pair_relative_velocity, dim=-1
    )
    pair_features = torch.stack(
        [
            (pair_distance / 30.0).clamp(max=3.0),
            (pair_closing_speed / 15.0).clamp(-2.0, 2.0),
            torch.exp(-pair_distance / 4.0),
            (pair_speed / 15.0).clamp(max=3.0),
        ],
        dim=-1,
    )
    mean_speed = torch.linalg.vector_norm(
        velocity, dim=-1
    ).mean(dim=-1, keepdim=True)
    motion_feature = (mean_speed / 20.0).clamp(max=3.0)

    traffic = input_dict.get("traffic_light_nodes")
    traffic_mask = input_dict.get("traffic_light_mask")
    if traffic is None or traffic_mask is None:
        traffic_features = trajectories.new_zeros(
            batch_size, num_modes, num_horizons, 3
        )
    else:
        traffic = traffic.to(device=device, dtype=trajectories.dtype)
        traffic_mask = traffic_mask.to(device=device).bool()
        traffic_query = position.reshape(
            batch_size, num_modes * num_horizons * 2, 2
        )
        traffic_distance = torch.cdist(
            traffic_query, traffic[..., 0:2]
        ).reshape(
            batch_size, num_modes, num_horizons, 2, traffic.shape[1]
        )
        stop_mask = traffic_mask & (
            (traffic[..., 4] > 0.5) | (traffic[..., 5] > 0.5)
        )
        go_mask = traffic_mask & (traffic[..., 6] > 0.5)
        stop_distance = traffic_distance.masked_fill(
            ~stop_mask[:, None, None, None],
            torch.finfo(trajectories.dtype).max,
        ).amin(dim=(-1, -2))
        go_distance = traffic_distance.masked_fill(
            ~go_mask[:, None, None, None],
            torch.finfo(trajectories.dtype).max,
        ).amin(dim=(-1, -2))
        stop_distance = torch.where(
            stop_mask.any(dim=-1)[:, None, None],
            stop_distance,
            stop_distance.new_full(stop_distance.shape, 50.0),
        )
        go_distance = torch.where(
            go_mask.any(dim=-1)[:, None, None],
            go_distance,
            go_distance.new_full(go_distance.shape, 50.0),
        )
        traffic_features = torch.stack(
            [
                (stop_distance / 50.0).clamp(max=2.0),
                torch.exp(-stop_distance / 8.0),
                torch.exp(-go_distance / 8.0),
            ],
            dim=-1,
        )

    evidence = torch.cat(
        [
            map_features,
            neighbor_features,
            pair_features,
            motion_feature,
            traffic_features,
        ],
        dim=-1,
    )
    if evidence.shape[-1] != SPATIAL_EVIDENCE_DIM:
        raise RuntimeError(
            f"Spatial evidence has {evidence.shape[-1]} features, "
            f"expected {SPATIAL_EVIDENCE_DIM}"
        )
    return torch.nan_to_num(
        evidence, nan=0.0, posinf=4.0, neginf=-4.0
    ).to(dtype=canonical_trajectories.dtype)
