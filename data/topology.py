"""Inference-time lane topology and regulatory evidence for Waymo scenes."""

from collections import Counter, defaultdict
import math

import numpy as np


RELATION_PAD = 0
RELATION_SAME_LANE_PREV = 1
RELATION_SAME_LANE_NEXT = 2
RELATION_PREDECESSOR = 3
RELATION_SUCCESSOR = 4


def _sampled_chunks(
    polylines,
    point_sampled_interval,
    vector_break_dist_thresh,
    num_points_each_polyline,
):
    """Reproduce WaymoDataset.generate_batch_polylines_from_map chunking."""
    sampled_indices = np.arange(
        0, len(polylines), point_sampled_interval, dtype=np.int64
    )
    sampled_points = polylines[sampled_indices]
    shifted_points = np.roll(sampled_points, shift=1, axis=0)
    shifted_points[0] = sampled_points[0]
    break_indices = np.nonzero(
        np.linalg.norm(
            sampled_points[:, 0:2] - shifted_points[:, 0:2], axis=-1
        )
        > vector_break_dist_thresh
    )[0]
    sampled_segments = np.array_split(sampled_indices, break_indices)

    chunks = []
    for segment in sampled_segments:
        if len(segment) == 0:
            continue
        for offset in range(0, len(segment), num_points_each_polyline):
            chunks.append(segment[offset : offset + num_points_each_polyline])
    return chunks


def build_map_chunk_metadata(
    map_infos,
    point_sampled_interval=1,
    vector_break_dist_thresh=1.0,
    num_points_each_polyline=20,
):
    """Attach raw lane identity and graph relations to preprocessing chunks."""
    polylines = np.asarray(map_infos["all_polylines"], dtype=np.float32)
    chunks = _sampled_chunks(
        polylines,
        int(point_sampled_interval),
        float(vector_break_dist_thresh),
        int(num_points_each_polyline),
    )

    point_lane_id = np.full(len(polylines), -1, dtype=np.int64)
    lane_records = {}
    for lane in map_infos.get("lane", []):
        lane_id = int(lane["id"])
        start, end = lane["polyline_index"]
        point_lane_id[int(start) : int(end)] = lane_id
        lane_records[lane_id] = lane

    chunk_lane_id = np.full(len(chunks), -1, dtype=np.int64)
    chunk_first_point = np.full(len(chunks), -1, dtype=np.int64)
    lane_chunks = defaultdict(list)
    for chunk_id, point_indices in enumerate(chunks):
        chunk_first_point[chunk_id] = int(point_indices[0])
        lane_values = point_lane_id[point_indices]
        lane_values = lane_values[lane_values >= 0]
        if lane_values.size:
            lane_id = int(Counter(lane_values.tolist()).most_common(1)[0][0])
            chunk_lane_id[chunk_id] = lane_id
            lane_chunks[lane_id].append(chunk_id)

    chunk_lane_ordinal = np.full(len(chunks), -1, dtype=np.int64)
    chunk_lane_count = np.zeros(len(chunks), dtype=np.int64)
    chunk_speed_limit_mph = np.zeros(len(chunks), dtype=np.float32)
    for lane_id, chunk_ids in lane_chunks.items():
        chunk_ids.sort(key=lambda value: int(chunk_first_point[value]))
        count = len(chunk_ids)
        speed_limit = float(lane_records[lane_id].get("speed_limit_mph", 0.0))
        for ordinal, chunk_id in enumerate(chunk_ids):
            chunk_lane_ordinal[chunk_id] = ordinal
            chunk_lane_count[chunk_id] = count
            chunk_speed_limit_mph[chunk_id] = speed_limit

    adjacency = defaultdict(set)
    for chunk_ids in lane_chunks.values():
        for previous, following in zip(chunk_ids[:-1], chunk_ids[1:]):
            adjacency[following].add((previous, RELATION_SAME_LANE_PREV))
            adjacency[previous].add((following, RELATION_SAME_LANE_NEXT))

    for lane_id, lane in lane_records.items():
        current_chunks = lane_chunks.get(lane_id, [])
        if not current_chunks:
            continue
        current_first = current_chunks[0]
        current_last = current_chunks[-1]
        for predecessor_id in lane.get("entry_lanes", []):
            predecessor_chunks = lane_chunks.get(int(predecessor_id), [])
            if predecessor_chunks:
                adjacency[current_first].add(
                    (predecessor_chunks[-1], RELATION_PREDECESSOR)
                )
        for successor_id in lane.get("exit_lanes", []):
            successor_chunks = lane_chunks.get(int(successor_id), [])
            if successor_chunks:
                adjacency[current_last].add(
                    (successor_chunks[0], RELATION_SUCCESSOR)
                )

    return {
        "chunk_lane_id": chunk_lane_id,
        "chunk_segment_id": np.arange(len(chunks), dtype=np.int64),
        "chunk_lane_ordinal": chunk_lane_ordinal,
        "chunk_lane_count": chunk_lane_count,
        "chunk_speed_limit_mph": chunk_speed_limit_mph,
        "adjacency": adjacency,
        "lane_records": lane_records,
        "num_chunks": len(chunks),
    }


def select_map_chunk_metadata(metadata, selected_chunk_ids, max_neighbors=8):
    """Map scene-global chunk relations into each center view's local indices."""
    selected_chunk_ids = np.asarray(selected_chunk_ids, dtype=np.int64)
    num_views, num_polylines = selected_chunk_ids.shape
    lane_id = metadata["chunk_lane_id"][selected_chunk_ids]
    segment_id = metadata["chunk_segment_id"][selected_chunk_ids]
    lane_ordinal = metadata["chunk_lane_ordinal"][selected_chunk_ids]
    lane_count = metadata["chunk_lane_count"][selected_chunk_ids]
    speed_limit = metadata["chunk_speed_limit_mph"][selected_chunk_ids]

    neighbor_index = np.full(
        (num_views, num_polylines, max_neighbors), -1, dtype=np.int64
    )
    neighbor_type = np.zeros(
        (num_views, num_polylines, max_neighbors), dtype=np.int64
    )
    neighbor_mask = np.zeros(
        (num_views, num_polylines, max_neighbors), dtype=bool
    )
    truncated = 0
    for view_idx in range(num_views):
        local_by_global = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(selected_chunk_ids[view_idx])
        }
        for local_idx, global_idx in enumerate(selected_chunk_ids[view_idx]):
            local_neighbors = []
            for neighbor_global, relation in metadata["adjacency"].get(
                int(global_idx), ()
            ):
                neighbor_local = local_by_global.get(int(neighbor_global))
                if neighbor_local is not None:
                    local_neighbors.append((int(relation), int(neighbor_local)))
            local_neighbors = sorted(set(local_neighbors))
            truncated += max(0, len(local_neighbors) - max_neighbors)
            for slot, (relation, neighbor_local) in enumerate(
                local_neighbors[:max_neighbors]
            ):
                neighbor_index[view_idx, local_idx, slot] = neighbor_local
                neighbor_type[view_idx, local_idx, slot] = relation
                neighbor_mask[view_idx, local_idx, slot] = True

    return {
        "map_lane_id": lane_id,
        "map_segment_id": segment_id,
        "map_lane_segment_ordinal": lane_ordinal,
        "map_lane_segment_count": lane_count,
        "map_speed_limit_mph": speed_limit,
        "map_topology_neighbor_index": neighbor_index,
        "map_topology_neighbor_type": neighbor_type,
        "map_topology_neighbor_mask": neighbor_mask,
        "topology_neighbor_truncated": np.asarray([truncated], dtype=np.int64),
    }


def _traffic_state_one_hot(state):
    state_name = str(state)
    output = np.zeros(4, dtype=np.float32)
    if "STOP" in state_name:
        output[1] = 1.0
    elif "CAUTION" in state_name:
        output[2] = 1.0
    elif "GO" in state_name:
        output[3] = 1.0
    else:
        output[0] = 1.0
    return output


def _to_local(point, center):
    delta = np.asarray(point, dtype=np.float32) - center[0:3]
    cos_h = math.cos(float(center[6]))
    sin_h = math.sin(float(center[6]))
    return np.asarray(
        [
            cos_h * delta[0] + sin_h * delta[1],
            -sin_h * delta[0] + cos_h * delta[1],
            delta[2],
        ],
        dtype=np.float32,
    )


def build_regulatory_evidence(
    info,
    center_objects,
    selected_lane_ids,
    max_tokens=64,
    max_controlled_segments=64,
):
    """Build exact lane-linked traffic-light and stop-sign tokens per view."""
    entries = []
    dynamic = info.get("dynamic_map_infos", {})
    current_idx = int(info["current_time_index"])
    try:
        lane_ids = np.asarray(dynamic.get("lane_id", [])[current_idx]).reshape(-1)
        states = np.asarray(dynamic.get("state", [])[current_idx]).reshape(-1)
        stop_points = np.asarray(
            dynamic.get("stop_point", [])[current_idx], dtype=np.float32
        ).reshape(-1, 3)
    except (IndexError, TypeError, ValueError):
        lane_ids = np.zeros(0, dtype=np.int64)
        states = np.zeros(0, dtype=object)
        stop_points = np.zeros((0, 3), dtype=np.float32)
    for lane_id, state, stop_point in zip(lane_ids, states, stop_points):
        entries.append(
            {
                "lane_id": int(lane_id),
                "position": np.asarray(stop_point, dtype=np.float32),
                "state": _traffic_state_one_hot(state),
                "is_traffic": 1.0,
                "is_stop_sign": 0.0,
            }
        )

    for stop_sign in info.get("map_infos", {}).get("stop_sign", []):
        position = np.asarray(stop_sign["position"], dtype=np.float32)
        for lane_id in stop_sign.get("lane_ids", []):
            entries.append(
                {
                    "lane_id": int(lane_id),
                    "position": position,
                    "state": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "is_traffic": 0.0,
                    "is_stop_sign": 1.0,
                }
            )

    num_views = len(center_objects)
    features = np.zeros((num_views, max_tokens, 9), dtype=np.float32)
    mask = np.zeros((num_views, max_tokens), dtype=bool)
    regulatory_lane_id = np.full((num_views, max_tokens), -1, dtype=np.int64)
    map_index = np.full(
        (num_views, max_tokens, max_controlled_segments), -1, dtype=np.int64
    )
    map_index_mask = np.zeros_like(map_index, dtype=bool)
    token_truncated = 0
    segment_truncated = 0

    for view_idx, center in enumerate(center_objects):
        selected_lanes = set(
            int(value) for value in selected_lane_ids[view_idx] if int(value) >= 0
        )
        ranked_entries = []
        for entry_idx, entry in enumerate(entries):
            local_position = _to_local(entry["position"], center)
            controlled_selected = int(entry["lane_id"]) in selected_lanes
            distance = float(np.linalg.norm(local_position[0:2]))
            ranked_entries.append(
                (not controlled_selected, distance, entry_idx, local_position, entry)
            )
        ranked_entries.sort(key=lambda value: value[:3])
        token_truncated += max(0, len(ranked_entries) - max_tokens)
        for token_idx, (_, _, _, local_position, entry) in enumerate(
            ranked_entries[:max_tokens]
        ):
            lane = int(entry["lane_id"])
            features[view_idx, token_idx, 0:3] = local_position
            features[view_idx, token_idx, 3:7] = entry["state"]
            features[view_idx, token_idx, 7] = entry["is_traffic"]
            features[view_idx, token_idx, 8] = entry["is_stop_sign"]
            mask[view_idx, token_idx] = True
            regulatory_lane_id[view_idx, token_idx] = lane
            controlled = np.flatnonzero(selected_lane_ids[view_idx] == lane)
            segment_truncated += max(0, len(controlled) - max_controlled_segments)
            count = min(len(controlled), max_controlled_segments)
            if count:
                map_index[view_idx, token_idx, :count] = controlled[:count]
                map_index_mask[view_idx, token_idx, :count] = True

    lane_records = info.get("map_infos", {}).get("lane", [])
    has_merge = any(len(lane.get("entry_lanes", [])) > 1 for lane in lane_records)
    has_branch = any(len(lane.get("exit_lanes", [])) > 1 for lane in lane_records)
    scene_flags = np.asarray(
        [[
            any(entry["is_traffic"] > 0 for entry in entries),
            any(entry["is_stop_sign"] > 0 for entry in entries),
            has_merge,
            has_branch,
            has_merge or has_branch,
        ]],
        dtype=bool,
    )
    return {
        "regulatory_features": features,
        "regulatory_mask": mask,
        "regulatory_lane_id": regulatory_lane_id,
        "regulatory_map_index": map_index,
        "regulatory_map_index_mask": map_index_mask,
        "regulatory_token_truncated": np.asarray([token_truncated], dtype=np.int64),
        "regulatory_segment_truncated": np.asarray(
            [segment_truncated], dtype=np.int64
        ),
        "scene_flags": scene_flags,
        "raw_traffic_count": np.asarray(
            [sum(entry["is_traffic"] > 0 for entry in entries)], dtype=np.int64
        ),
        "raw_stop_sign_lane_count": np.asarray(
            [sum(entry["is_stop_sign"] > 0 for entry in entries)], dtype=np.int64
        ),
    }
