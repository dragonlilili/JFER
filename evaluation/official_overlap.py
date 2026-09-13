"""Waymo 2025 interaction-prediction overlap-rate evaluation.

This module implements the challenge definition that cannot be recovered from
the two-agent tensors used by the base evaluator: the top-confidence
joint mode is checked against every object visible at prediction time and the
jointly predicted agents are checked against one another.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


HORIZON_STEPS = {"3s": 6, "5s": 10, "8s": 16}
OBJECT_TYPES = ("VEHICLE", "PEDESTRIAN", "CYCLIST")
_TYPE_PRIORITY = {
    "TYPE_UNSET": 0,
    "TYPE_OTHER": 1,
    "TYPE_VEHICLE": 2,
    "TYPE_PEDESTRIAN": 3,
    "TYPE_CYCLIST": 4,
}


def _as_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _canonical_object_type(value: Any) -> str:
    value = str(_as_scalar(value)).upper()
    if not value.startswith("TYPE_"):
        value = f"TYPE_{value}"
    if value not in _TYPE_PRIORITY:
        raise ValueError(f"Unsupported Waymo object type: {value}")
    return value


def get_pair_object_type(object_types: Sequence[Any]) -> str:
    """Return the least-common object type used for Waymo joint buckets."""
    if len(object_types) != 2:
        raise ValueError(f"Expected two object types, got {len(object_types)}")
    selected = max(
        (_canonical_object_type(value) for value in object_types),
        key=_TYPE_PRIORITY.__getitem__,
    )
    short_name = selected.removeprefix("TYPE_")
    if short_name not in OBJECT_TYPES:
        raise ValueError(f"Unsupported interaction pair type: {selected}")
    return short_name


def prediction_headings(positions: np.ndarray) -> np.ndarray:
    """Match Waymo's heading construction from adjacent predicted positions."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError(f"Expected [T, 2] positions with T >= 2, got {positions.shape}")

    forward = positions[1:] - positions[:-1]
    segment_heading = np.arctan2(forward[:, 1], forward[:, 0])
    headings = np.empty(positions.shape[0], dtype=np.float64)
    headings[0] = segment_heading[0]
    headings[-1] = segment_heading[-1]
    if positions.shape[0] > 2:
        prev_heading = segment_heading[:-1]
        next_heading = segment_heading[1:]
        headings[1:-1] = np.arctan2(
            np.sin(prev_heading) + np.sin(next_heading),
            np.cos(prev_heading) + np.cos(next_heading),
        )
    return headings


def oriented_boxes_overlap(
    center_a: np.ndarray,
    heading_a: np.ndarray,
    length_a: np.ndarray,
    width_a: np.ndarray,
    center_b: np.ndarray,
    heading_b: np.ndarray,
    length_b: np.ndarray,
    width_b: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Return positive-area intersections for broadcastable oriented boxes.

    The separating-axis test uses the two local axes from each rectangle.
    Strict inequalities intentionally exclude boxes that only touch, matching
    Waymo's positive-intersection-area requirement.
    """
    center_a = np.asarray(center_a, dtype=np.float64)
    center_b = np.asarray(center_b, dtype=np.float64)
    heading_a = np.asarray(heading_a, dtype=np.float64)
    heading_b = np.asarray(heading_b, dtype=np.float64)
    length_a = np.asarray(length_a, dtype=np.float64)
    width_a = np.asarray(width_a, dtype=np.float64)
    length_b = np.asarray(length_b, dtype=np.float64)
    width_b = np.asarray(width_b, dtype=np.float64)

    a_u = np.stack((np.cos(heading_a), np.sin(heading_a)), axis=-1)
    a_v = np.stack((-np.sin(heading_a), np.cos(heading_a)), axis=-1)
    b_u = np.stack((np.cos(heading_b), np.sin(heading_b)), axis=-1)
    b_v = np.stack((-np.sin(heading_b), np.cos(heading_b)), axis=-1)
    delta = center_b - center_a

    def dot(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.sum(lhs * rhs, axis=-1)

    a_half_l = length_a * 0.5
    a_half_w = width_a * 0.5
    b_half_l = length_b * 0.5
    b_half_w = width_b * 0.5

    au_bu = np.abs(dot(a_u, b_u))
    au_bv = np.abs(dot(a_u, b_v))
    av_bu = np.abs(dot(a_v, b_u))
    av_bv = np.abs(dot(a_v, b_v))

    overlaps = (
        (np.abs(dot(delta, a_u)) < a_half_l + b_half_l * au_bu + b_half_w * au_bv - epsilon)
        & (np.abs(dot(delta, a_v)) < a_half_w + b_half_l * av_bu + b_half_w * av_bv - epsilon)
        & (np.abs(dot(delta, b_u)) < b_half_l + a_half_l * au_bu + a_half_w * av_bu - epsilon)
        & (np.abs(dot(delta, b_v)) < b_half_w + a_half_l * au_bv + a_half_w * av_bv - epsilon)
    )

    finite = (
        np.all(np.isfinite(center_a), axis=-1)
        & np.all(np.isfinite(center_b), axis=-1)
        & np.isfinite(heading_a)
        & np.isfinite(heading_b)
        & np.isfinite(length_a)
        & np.isfinite(width_a)
        & np.isfinite(length_b)
        & np.isfinite(width_b)
    )
    positive_dimensions = (
        (length_a > 0.0)
        & (width_a > 0.0)
        & (length_b > 0.0)
        & (width_b > 0.0)
    )
    return overlaps & finite & positive_dimensions


def _sample_official_prediction_steps(pred_trajs: np.ndarray) -> np.ndarray:
    pred_trajs = np.asarray(pred_trajs, dtype=np.float64)
    if pred_trajs.ndim != 3 or pred_trajs.shape[0] != 2 or pred_trajs.shape[-1] != 2:
        raise ValueError(f"Expected joint trajectory [2, T, 2], got {pred_trajs.shape}")
    if pred_trajs.shape[1] >= 80:
        sampled = pred_trajs[:, 4:80:5, :]
    elif pred_trajs.shape[1] == 16:
        sampled = pred_trajs
    else:
        raise ValueError(
            "Expected either 80 frames at 10 Hz or 16 official frames at 2 Hz, "
            f"got {pred_trajs.shape[1]} frames"
        )
    if sampled.shape[1] != 16:
        raise ValueError(f"Expected 16 official prediction steps, got {sampled.shape}")
    return sampled


def evaluate_scenario_overlap(
    prediction: Mapping[str, Any], scenario: Mapping[str, Any]
) -> Dict[str, Any]:
    """Evaluate one multi-modal joint prediction under the 2025 definition."""
    scenario_id = str(_as_scalar(prediction["scenario_id"]))
    if str(scenario["scenario_id"]) != scenario_id:
        raise ValueError(
            f"Scenario mismatch: prediction={scenario_id}, data={scenario['scenario_id']}"
        )

    pred_scores = np.asarray(prediction["pred_scores"], dtype=np.float64)
    all_modes = np.asarray(prediction["pred_trajs"], dtype=np.float64)
    if pred_scores.ndim != 1 or all_modes.shape[0] != pred_scores.shape[0]:
        raise ValueError(
            f"Incompatible scores {pred_scores.shape} and trajectories {all_modes.shape}"
        )
    top_mode_index = int(np.nanargmax(pred_scores))
    pred_positions = _sample_official_prediction_steps(all_modes[top_mode_index])
    pred_headings = np.stack(
        [prediction_headings(pred_positions[agent_idx]) for agent_idx in range(2)]
    )

    track_infos = scenario["track_infos"]
    all_track_ids = np.asarray(track_infos["object_id"])
    all_track_types = list(track_infos["object_type"])
    all_trajs = np.asarray(track_infos["trajs"], dtype=np.float64)
    if all_trajs.ndim != 3 or all_trajs.shape[-1] < 10:
        raise ValueError(f"Unexpected scenario trajectory shape: {all_trajs.shape}")

    target_ids = np.asarray(prediction["object_id"])
    if target_ids.shape != (2,):
        raise ValueError(f"Expected two target object IDs, got {target_ids.shape}")
    id_to_index = {_as_scalar(object_id): idx for idx, object_id in enumerate(all_track_ids)}
    try:
        target_indices = np.asarray(
            [id_to_index[_as_scalar(object_id)] for object_id in target_ids], dtype=np.int64
        )
    except KeyError as exc:
        raise KeyError(f"Target object {exc.args[0]} missing in scenario {scenario_id}") from exc

    current_step = int(scenario.get("current_time_index", 10))
    future_steps = current_step + np.arange(1, 17, dtype=np.int64) * 5
    if future_steps[-1] >= all_trajs.shape[1]:
        raise ValueError(
            f"Scenario {scenario_id} has only {all_trajs.shape[1]} frames; "
            f"need index {future_steps[-1]}"
        )

    target_trajs = all_trajs[target_indices]
    target_dimensions = target_trajs[:, future_steps, 3:5]
    target_lengths = target_dimensions[:, :, 0]
    target_widths = target_dimensions[:, :, 1]

    scene_centers = all_trajs[:, future_steps, 0:2]
    scene_lengths = all_trajs[:, future_steps, 3]
    scene_widths = all_trajs[:, future_steps, 4]
    scene_headings = all_trajs[:, future_steps, 6]
    visible_now = all_trajs[:, current_step, 9] > 0.0
    valid_future = all_trajs[:, future_steps, 9] > 0.0

    scene_step_overlap = np.zeros(16, dtype=bool)
    for agent_idx, own_track_idx in enumerate(target_indices):
        collision = oriented_boxes_overlap(
            pred_positions[agent_idx][None, :, :],
            pred_headings[agent_idx][None, :],
            target_lengths[agent_idx][None, :],
            target_widths[agent_idx][None, :],
            scene_centers,
            scene_headings,
            scene_lengths,
            scene_widths,
        )
        collision &= visible_now[:, None] & valid_future
        collision[own_track_idx, :] = False
        scene_step_overlap |= np.any(collision, axis=0)

    pair_step_overlap = oriented_boxes_overlap(
        pred_positions[0],
        pred_headings[0],
        target_lengths[0],
        target_widths[0],
        pred_positions[1],
        pred_headings[1],
        target_lengths[1],
        target_widths[1],
    )
    combined_step_overlap = scene_step_overlap | pair_step_overlap

    pair_types = [all_track_types[idx] for idx in target_indices]
    pair_type = get_pair_object_type(pair_types)
    official_overlap = {
        horizon: bool(np.any(combined_step_overlap[:num_steps]))
        for horizon, num_steps in HORIZON_STEPS.items()
    }
    scene_overlap = {
        horizon: bool(np.any(scene_step_overlap[:num_steps]))
        for horizon, num_steps in HORIZON_STEPS.items()
    }
    pair_prediction_overlap = {
        horizon: bool(np.any(pair_step_overlap[:num_steps]))
        for horizon, num_steps in HORIZON_STEPS.items()
    }
    collision_indices = np.flatnonzero(combined_step_overlap)

    return {
        "scenario_id": scenario_id,
        "pair_type": pair_type,
        "object_ids": [_as_scalar(value) for value in target_ids.tolist()],
        "top_mode_index": top_mode_index,
        "top_confidence": float(pred_scores[top_mode_index]),
        "visible_object_count": int(np.count_nonzero(visible_now)),
        "official_overlap": official_overlap,
        "scene_gt_overlap": scene_overlap,
        "joint_prediction_overlap": pair_prediction_overlap,
        "first_overlap_seconds": (
            None if collision_indices.size == 0 else float((collision_indices[0] + 1) * 0.5)
        ),
    }


def aggregate_overlap_results(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    results = list(results)
    if not results:
        raise ValueError("No overlap results to aggregate")

    by_type: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in results:
        by_type[str(item["pair_type"])].append(item)

    cells: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for object_type in OBJECT_TYPES:
        cells[object_type] = {}
        type_rows = by_type.get(object_type, [])
        for horizon in HORIZON_STEPS:
            overlap_count = sum(bool(row["official_overlap"][horizon]) for row in type_rows)
            scene_count = sum(bool(row["scene_gt_overlap"][horizon]) for row in type_rows)
            pair_count = sum(
                bool(row["joint_prediction_overlap"][horizon]) for row in type_rows
            )
            count = len(type_rows)
            cells[object_type][horizon] = {
                "count": count,
                "overlap_count": overlap_count,
                "overlap_rate": None if count == 0 else overlap_count / count,
                "scene_gt_overlap_count": scene_count,
                "joint_prediction_overlap_count": pair_count,
            }

    by_object_type = {}
    for object_type in OBJECT_TYPES:
        rates = [
            cells[object_type][horizon]["overlap_rate"] for horizon in HORIZON_STEPS
        ]
        rates = [rate for rate in rates if rate is not None]
        by_object_type[object_type] = None if not rates else float(np.mean(rates))

    by_horizon = {}
    for horizon in HORIZON_STEPS:
        rates = [cells[object_type][horizon]["overlap_rate"] for object_type in OBJECT_TYPES]
        rates = [rate for rate in rates if rate is not None]
        by_horizon[horizon] = None if not rates else float(np.mean(rates))

    cell_rates = [
        cells[object_type][horizon]["overlap_rate"]
        for object_type in OBJECT_TYPES
        for horizon in HORIZON_STEPS
        if cells[object_type][horizon]["overlap_rate"] is not None
    ]
    all_flags = [
        bool(row["official_overlap"][horizon])
        for row in results
        for horizon in HORIZON_STEPS
    ]
    return {
        "definition": "Waymo 2025 Interaction Prediction Overlap Rate",
        "scenario_count": len(results),
        "pair_type_counts": {key: len(by_type.get(key, [])) for key in OBJECT_TYPES},
        "per_type_horizon": cells,
        "by_object_type": by_object_type,
        "by_horizon": by_horizon,
        "overall_macro": float(np.mean(cell_rates)),
        "overall_micro": float(np.mean(all_flags)),
        "mean_visible_object_count": float(
            np.mean([row["visible_object_count"] for row in results])
        ),
    }
