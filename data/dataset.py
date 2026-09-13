# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from data.womd_dataset import WaymoDataset
from evaluation.womd_evaluation import (
    _default_metrics_config,
    object_type_to_id,
)
from utils import common as common_utils
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util


class WaymoInteractiveDataset(WaymoDataset):
    """Pair-centric WOMD dataset for real interaction prediction.

    Each item is one interacting pair. The scene is encoded once in the first
    target agent's local frame, and the model predicts joint futures for both
    target agents: [num_modes, 2, future_steps, 2].
    """

    def filter_info_by_interactive_pair(self, infos, valid_object_types=None):
        selection = str(
            self.dataset_cfg.get("PAIR_SELECTION", "closest")
        ).lower()
        if selection in {
            "objects_of_interest",
            "official_objects_of_interest",
            "official_ooi",
        }:
            self.logger.info(
                "Bypass filter_info_by_interactive_pair because official "
                "objects_of_interest filtering is enabled"
            )
            return infos
        valid_object_types = set(valid_object_types or self.dataset_cfg.OBJECT_TYPE)
        ret_infos = []
        for cur_info in infos:
            target_types = cur_info["tracks_to_predict"]["object_type"]
            valid_count = sum(t in valid_object_types for t in target_types)
            if valid_count >= 2:
                ret_infos.append(cur_info)
        self.logger.info(
            f"Total scenes after filter_info_by_interactive_pair: {len(ret_infos)}"
        )
        return ret_infos

    def filter_info_by_objects_of_interest(self, infos, valid_object_types=None):
        """Keep only official WOMD interaction-prediction training pairs.

        ``objects_of_interest`` stores object ids, while
        ``tracks_to_predict`` stores track indices. The lightweight info files
        do not contain the complete object-id table, so this filter performs
        the checks available at index-load time and ``_select_pair_indices``
        performs the exact id-to-track validation when the scenario is read.
        """
        valid_object_types = set(valid_object_types or self.dataset_cfg.OBJECT_TYPE)
        ret_infos = []
        malformed = 0
        invalid_target_types = 0
        for cur_info in infos:
            objects_of_interest = np.asarray(
                cur_info.get("objects_of_interest", []), dtype=object
            ).reshape(-1)
            if (
                objects_of_interest.size != 2
                or len({str(value) for value in objects_of_interest}) != 2
            ):
                malformed += 1
                continue

            target_types = cur_info.get("tracks_to_predict", {}).get(
                "object_type", []
            )
            if sum(obj_type in valid_object_types for obj_type in target_types) < 2:
                invalid_target_types += 1
                continue
            ret_infos.append(cur_info)

        self.logger.info(
            "Total scenes after filter_info_by_objects_of_interest: "
            f"{len(ret_infos)} (dropped_no_exact_pair={malformed}, "
            f"dropped_invalid_target_types={invalid_target_types})"
        )
        return ret_infos

    def _select_objects_of_interest_pair(
        self, info, obj_trajs_full, obj_types, current_time_index
    ):
        objects_of_interest = np.asarray(
            info.get("objects_of_interest", []), dtype=object
        ).reshape(-1)
        if (
            objects_of_interest.size != 2
            or len({str(value) for value in objects_of_interest}) != 2
        ):
            raise RuntimeError(
                f"Scenario {info['scenario_id']} does not contain exactly two "
                "distinct objects_of_interest"
            )

        object_ids = np.asarray(info["track_infos"]["object_id"], dtype=object)
        object_id_to_index = {
            str(object_id): idx for idx, object_id in enumerate(object_ids)
        }
        missing = [
            object_id
            for object_id in objects_of_interest
            if str(object_id) not in object_id_to_index
        ]
        if missing:
            raise RuntimeError(
                f"Scenario {info['scenario_id']} is missing objects_of_interest "
                f"from track_infos: {missing}"
            )

        pair_indices = np.asarray(
            [object_id_to_index[str(object_id)] for object_id in objects_of_interest],
            dtype=np.int64,
        )
        target_indices = {
            int(index)
            for index in info["tracks_to_predict"]["track_index"]
        }
        if any(int(index) not in target_indices for index in pair_indices):
            raise RuntimeError(
                f"Scenario {info['scenario_id']} has an object_of_interest that "
                "is not in tracks_to_predict"
            )

        valid_types = set(self.dataset_cfg.get("OBJECT_TYPE", []))
        invalid_types = [
            str(obj_types[int(index)])
            for index in pair_indices
            if obj_types[int(index)] not in valid_types
        ]
        if invalid_types:
            raise RuntimeError(
                f"Scenario {info['scenario_id']} has unsupported official pair "
                f"types: {invalid_types}"
            )
        if any(
            obj_trajs_full[int(index), current_time_index, -1] <= 0
            for index in pair_indices
        ):
            raise RuntimeError(
                f"Scenario {info['scenario_id']} has an official pair member "
                "without a valid current state"
            )
        return pair_indices

    def _select_pair_indices(self, info, obj_trajs_full, obj_types, current_time_index):
        selection = str(
            self.dataset_cfg.get("PAIR_SELECTION", "closest")
        ).lower()
        if selection in {
            "objects_of_interest",
            "official_objects_of_interest",
            "official_ooi",
        }:
            return self._select_objects_of_interest_pair(
                info, obj_trajs_full, obj_types, current_time_index
            )

        target_indices = np.asarray(info["tracks_to_predict"]["track_index"], dtype=np.int64)
        valid_types = set(self.dataset_cfg.get("OBJECT_TYPE", []))
        valid_target_indices = [
            int(idx)
            for idx in target_indices
            if obj_types[int(idx)] in valid_types
            and obj_trajs_full[int(idx), current_time_index, -1] > 0
        ]
        if len(valid_target_indices) < 2:
            raise RuntimeError(
                f"Scenario {info['scenario_id']} has fewer than two valid targets"
            )
        if len(valid_target_indices) == 2:
            return np.asarray(valid_target_indices, dtype=np.int64)

        # Normal WOMD scenes may contain more than two targets. For joint
        # training, mine the pair whose constant-velocity extrapolations have
        # the smallest closest-approach distance. This uses history/current
        # state only and is a better interaction prior than current distance.
        current = obj_trajs_full[
            valid_target_indices, current_time_index
        ]
        cur_xy = current[:, 0:2]
        cur_vel = current[:, 7:9]
        horizon = float(
            self.dataset_cfg.get("PAIR_INTERACTION_HORIZON", 8.0)
        )
        best_pair = None
        best_score = float("inf")
        for i in range(len(valid_target_indices)):
            for j in range(i + 1, len(valid_target_indices)):
                relative_position = cur_xy[j] - cur_xy[i]
                current_dist = float(np.linalg.norm(relative_position))
                if selection == "interaction_ttc":
                    relative_velocity = cur_vel[j] - cur_vel[i]
                    velocity_sq = float(
                        np.dot(relative_velocity, relative_velocity)
                    )
                    if velocity_sq > 1e-4:
                        closest_t = float(
                            np.clip(
                                -np.dot(
                                    relative_position, relative_velocity
                                ) / velocity_sq,
                                0.0,
                                horizon,
                            )
                        )
                    else:
                        closest_t = 0.0
                    closest_dist = float(
                        np.linalg.norm(
                            relative_position
                            + closest_t * relative_velocity
                        )
                    )
                    score = closest_dist + 0.1 * current_dist
                else:
                    score = current_dist
                if score < best_score:
                    best_score = score
                    best_pair = (valid_target_indices[i], valid_target_indices[j])
        return np.asarray(best_pair, dtype=np.int64)

    def _build_pair_current_state(self, obj_trajs_data, pair_filtered_indices):
        # Waymo agent feature layout:
        # [box 0:6], [type/center/sdc 6:11], [time 11:23],
        # [sin/cos heading 23:25], [vel 25:27], [accel 27:29].
        current_feat = obj_trajs_data[0, pair_filtered_indices, -1]
        pair_state = np.zeros((1, 2, 6), dtype=np.float32)
        pair_state[0, :, 0:2] = current_feat[:, 0:2]
        pair_state[0, :, 2:4] = current_feat[:, 25:27]
        pair_state[0, :, 4:6] = current_feat[:, 23:25]
        return pair_state

    def _canonicalize_pair_indices(self, pair_indices, object_ids):
        """Give the two OOI agents a stable role without using future data."""
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        if not bool(
            self.dataset_cfg.get("PAIR_CANONICALIZE_OOI_ORDER", False)
        ):
            return pair_indices

        pair_ids = [str(object_ids[int(index)]) for index in pair_indices]
        order = sorted(range(2), key=lambda index: pair_ids[index])
        return pair_indices[np.asarray(order, dtype=np.int64)]

    @staticmethod
    def _build_pair_frame(obj_trajs_full, pair_indices, current_time_index):
        """Build one deterministic midpoint SE(2) frame for the OOI pair."""
        current = obj_trajs_full[
            np.asarray(pair_indices, dtype=np.int64), current_time_index
        ].astype(np.float32)
        frame = current[:1].copy()
        frame[0, 0:3] = current[:, 0:3].mean(axis=0)
        frame[0, 3:6] = current[:, 3:6].max(axis=0)

        delta = current[1, 0:2] - current[0, 0:2]
        if float(np.linalg.norm(delta)) > 1e-3:
            frame[0, 6] = np.arctan2(delta[1], delta[0])
        else:
            # The ID-canonical first agent gives a deterministic fallback for
            # the rare case where both centers are numerically coincident.
            frame[0, 6] = current[0, 6]
        frame[0, 7:9] = current[:, 7:9].mean(axis=0)
        frame[0, 9] = 1.0
        return frame

    @staticmethod
    def _build_traffic_light_nodes(info, anchor_state, max_nodes=16):
        """Return current traffic-control stop points in the anchor SE(2) frame."""
        nodes = np.zeros((1, max_nodes, 8), dtype=np.float32)
        mask = np.zeros((1, max_nodes), dtype=bool)
        dynamic = info.get("dynamic_map_infos", None)
        if not isinstance(dynamic, dict):
            return nodes, mask

        current_idx = int(info["current_time_index"])
        try:
            lane_ids = np.asarray(
                dynamic.get("lane_id", [])[current_idx]
            ).reshape(-1)
            states = np.asarray(
                dynamic.get("state", [])[current_idx]
            ).reshape(-1)
            stop_points = np.asarray(
                dynamic.get("stop_point", [])[current_idx], dtype=np.float32
            ).reshape(-1, 3)
        except (IndexError, TypeError, ValueError):
            return nodes, mask

        count = min(len(lane_ids), len(states), len(stop_points), max_nodes)
        if count == 0:
            return nodes, mask

        delta = stop_points[:count] - anchor_state[None, 0:3]
        heading = float(anchor_state[6])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        local_x = cos_h * delta[:, 0] + sin_h * delta[:, 1]
        local_y = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
        nodes[0, :count, 0] = local_x
        nodes[0, :count, 1] = local_y
        nodes[0, :count, 2] = delta[:, 2]

        for idx, state in enumerate(states[:count]):
            state_name = str(state)
            if "STOP" in state_name:
                state_idx = 1
            elif "CAUTION" in state_name:
                state_idx = 2
            elif "GO" in state_name:
                state_idx = 3
            else:
                state_idx = 0
            nodes[0, idx, 3 + state_idx] = 1.0
            nodes[0, idx, 7] = 1.0
        mask[0, :count] = True
        return nodes, mask

    def create_scene_level_data(self, index):
        info_meta = self.infos[index]
        scene_id = info_meta["scenario_id"]
        with open(self.data_path / f"sample_{scene_id}.pkl", "rb") as f:
            info = pickle.load(f)

        sdc_track_index = info["sdc_track_index"]
        current_time_index = info["current_time_index"]
        timestamps = np.asarray(
            info["timestamps_seconds"][: current_time_index + 1],
            dtype=np.float32,
        )

        track_infos = info["track_infos"]
        obj_types_raw = np.asarray(track_infos["object_type"])
        obj_ids_raw = np.asarray(track_infos["object_id"])
        obj_trajs_full = track_infos["trajs"]
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        pair_indices = self._select_pair_indices(
            info, obj_trajs_full, obj_types_raw, current_time_index
        )
        pair_indices = self._canonicalize_pair_indices(
            pair_indices, obj_ids_raw
        )
        center_track_index = np.asarray([pair_indices[0]], dtype=np.int64)
        if bool(self.dataset_cfg.get("PAIR_CANONICAL_FRAME", False)):
            center_objects = self._build_pair_frame(
                obj_trajs_full, pair_indices, current_time_index
            )
        else:
            center_objects, center_track_index = self.get_interested_agents(
                track_index_to_predict=center_track_index,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types_raw,
                scene_id=scene_id,
            )

        (
            obj_trajs_data,
            obj_trajs_mask,
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            center_gt_trajs,
            center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new,
            sdc_track_index_new,
            obj_types,
            obj_ids,
        ) = self.create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=center_track_index,
            sdc_track_index=sdc_track_index,
            timestamps=timestamps,
            obj_types=obj_types_raw,
            obj_ids=obj_ids_raw,
        )

        pair_object_ids = obj_ids_raw[pair_indices]
        pair_filtered_indices = []
        for object_id in pair_object_ids:
            matches = np.where(obj_ids == object_id)[0]
            if len(matches) == 0:
                raise RuntimeError(
                    f"Pair object id {object_id} was filtered out in {scene_id}"
                )
            pair_filtered_indices.append(int(matches[0]))
        pair_filtered_indices = np.asarray(pair_filtered_indices, dtype=np.int64)

        pair_gt_trajs = obj_trajs_future_state[0, pair_filtered_indices][None]
        pair_gt_trajs_mask = obj_trajs_future_mask[0, pair_filtered_indices][None]
        pair_gt_trajs[pair_gt_trajs_mask == 0] = 0

        ret_dict = {
            "scenario_id": np.asarray([scene_id]),
            "obj_trajs": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": track_index_to_predict_new,
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "obj_types": obj_types,
            "obj_ids": obj_ids,
            "center_objects_world": center_objects,
            "center_objects_id": pair_object_ids[:1],
            "center_objects_type": obj_types_raw[pair_indices[:1]],
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "center_gt_trajs_src": obj_trajs_full[pair_indices[:1]],
            "pair_partner_index": pair_filtered_indices[1:2],
            "pair_track_index_to_predict": pair_indices[None].astype(np.int64),
            "pair_object_ids": pair_object_ids[None].astype(np.int64),
            "pair_object_types": obj_types_raw[pair_indices][None],
            "pair_center_objects_world": obj_trajs_full[
                pair_indices, current_time_index
            ][None].astype(np.float32),
            "pair_frame_world": center_objects[None].astype(np.float32),
            "pair_current_state": self._build_pair_current_state(
                obj_trajs_data, pair_filtered_indices
            ),
            "pair_gt_trajs": pair_gt_trajs.astype(np.float32),
            "pair_gt_trajs_mask": pair_gt_trajs_mask.astype(bool),
            "pair_gt_trajs_src": obj_trajs_full[pair_indices][None].astype(np.float32),
            "pair_is_official_ooi": np.asarray(
                [
                    str(self.dataset_cfg.get("PAIR_SELECTION", "closest")).lower()
                    in {
                        "objects_of_interest",
                        "official_objects_of_interest",
                        "official_ooi",
                    }
                ],
                dtype=bool,
            ),
        }

        traffic_nodes, traffic_mask = self._build_traffic_light_nodes(
            info, center_objects[0]
        )
        ret_dict["traffic_light_nodes"] = traffic_nodes
        ret_dict["traffic_light_mask"] = traffic_mask

        if not self.dataset_cfg.get("WITHOUT_HDMAP", False):
            if info["map_infos"]["all_polylines"].__len__() == 0:
                info["map_infos"]["all_polylines"] = np.zeros((2, 7), dtype=np.float32)
                print(f"Warning: empty HDMap {scene_id}")

            (
                map_polylines_data,
                map_polylines_mask,
                map_polylines_center,
            ) = self.create_map_data_for_center_objects(
                center_objects=center_objects,
                map_infos=info["map_infos"],
                center_offset=self.dataset_cfg.get(
                    "CENTER_OFFSET_OF_MAP", (30.0, 0)
                ),
            )

            ret_dict["map_polylines"] = map_polylines_data
            ret_dict["map_polylines_mask"] = map_polylines_mask > 0
            ret_dict["map_polylines_center"] = map_polylines_center

        return ret_dict

    def collate_batch(self, batch_list):
        batch_size = len(batch_list)
        key_to_list = {
            key: [batch_list[bs_idx][key] for bs_idx in range(batch_size)]
            for key in batch_list[0].keys()
        }

        input_dict = {}
        padded_keys = {
            "obj_trajs",
            "obj_trajs_mask",
            "map_polylines",
            "map_polylines_mask",
            "map_polylines_center",
            "obj_trajs_pos",
            "obj_trajs_last_pos",
            "obj_trajs_future_state",
            "obj_trajs_future_mask",
        }
        string_keys = {
            "scenario_id",
            "obj_types",
            "obj_ids",
            "center_objects_type",
            "center_objects_id",
            "pair_object_types",
        }
        for key, val_list in key_to_list.items():
            if key in padded_keys:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in string_keys:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [1 for _ in batch_list]
        return {
            "batch_size": batch_size,
            "input_dict": input_dict,
            "batch_sample_count": batch_sample_count,
        }

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        input_dict = batch_dict["input_dict"]
        pred_scores = batch_dict["pred_scores"]
        pred_trajs = batch_dict["pred_trajs"]
        pair_center_world = input_dict["pair_center_objects_world"].type_as(pred_trajs)

        batch_size, num_modes, num_agents, num_timestamps, _ = pred_trajs.shape
        pair_frame_world = input_dict.get("pair_frame_world", None)
        if pair_frame_world is None:
            anchor_world = pair_center_world[:, 0]
        else:
            anchor_world = pair_frame_world[:, 0].type_as(pred_trajs)
        trajs_world = common_utils.rotate_points_along_z(
            pred_trajs.reshape(batch_size, num_modes * num_agents * num_timestamps, 2),
            anchor_world[:, 6],
        ).reshape(batch_size, num_modes, num_agents, num_timestamps, 2)
        trajs_world = trajs_world + anchor_world[:, None, None, None, 0:2]

        pred_dict_list = []
        for batch_idx in range(batch_size):
            prediction = {
                "scenario_id": input_dict["scenario_id"][batch_idx],
                "pred_trajs": trajs_world[batch_idx].detach().cpu().numpy(),
                "pred_scores": pred_scores[batch_idx].detach().cpu().numpy(),
                "gt_trajs": input_dict[
                    "pair_gt_trajs_src"
                ][batch_idx].cpu().numpy(),
                "object_id": input_dict[
                    "pair_object_ids"
                ][batch_idx].cpu().numpy(),
                "object_type": input_dict[
                    "pair_object_types"
                ][batch_idx].tolist(),
                "track_index_to_predict": input_dict[
                    "pair_track_index_to_predict"
                ][batch_idx].cpu().numpy(),
            }
            optional_score_fields = {
                "pmw_s2_set_probability": "pmw_s2_set_probability",
                "pmw_s2_conditional_probability": (
                    "pmw_s2_conditional_probability"
                ),
                "pmw_s2_horizon_scores": "pmw_s2_horizon_scores",
            }
            for output_key, prediction_key in optional_score_fields.items():
                if output_key in batch_dict:
                    prediction[prediction_key] = batch_dict[output_key][
                        batch_idx
                    ].detach().cpu().numpy()
            pred_dict_list.append(prediction)
        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method="waymo_interaction", **kwargs):
        normalize_scores = bool(
            self.dataset_cfg.get("NORMALIZE_INTERACTION_SCORES", True)
        )
        metric_results, result_str = waymo_interaction_evaluation(
            pred_dicts, normalize_scores=normalize_scores
        )
        if bool(self.dataset_cfg.get("COMPUTE_SOFT_MAP", False)):
            # The installed Waymo op predates the 2025 leaderboard's Soft mAP
            # output. This implementation is separately checked against the
            # same op's hard mAP before its Soft mAP is used for selection.
            from tools.analyze_waymo_interaction_predictions import evaluate

            soft_result = evaluate(
                pred_dicts,
                progress_interval=int(
                    self.dataset_cfg.get("SOFT_MAP_PROGRESS_INTERVAL", 0)
                ),
                normalize_scores=normalize_scores,
            )
            metric_results.update(
                {
                    "soft_mAP": float(soft_result["soft_map"]),
                    "soft_mAP_ceiling": float(
                        soft_result[
                            "perfect_ranking_soft_map_ceiling"
                        ]
                    ),
                    "soft_mAP_gap": float(
                        soft_result["soft_map_gap_to_candidate_ceiling"]
                    ),
                    "soft_mAP_hard_consistency_error": abs(
                        float(soft_result["hard_map"])
                        - float(metric_results["mAP"])
                    ),
                }
            )
            for horizon, values in soft_result["by_horizon"].items():
                metric_results[f"soft_mAP/{horizon}"] = float(
                    values["soft_map"]
                )
            for object_type, values in soft_result[
                "by_object_type"
            ].items():
                metric_results[f"soft_mAP/{object_type}"] = float(
                    values["soft_map"]
                )
            result_str += (
                f", Soft-mAP={metric_results['soft_mAP']:.6f}, "
                f"Soft-mAP-ceiling={metric_results['soft_mAP_ceiling']:.6f}, "
                "hard-consistency-error="
                f"{metric_results['soft_mAP_hard_consistency_error']:.6f}"
            )
        metric_result_str = "\n"
        for key, value in metric_results.items():
            metric_result_str += f"{key}: {value:.4f} \n"
        metric_result_str += "\n" + result_str
        return metric_result_str, metric_results


class WaymoInteractivePairCenterDataset(WaymoInteractiveDataset):
    """Interaction dataset that encodes both agents in a pair as center objects.

    The sample remains pair-level for evaluation, but the model receives two
    center-object views per sample. This allows a pretrained single-agent base
    decoder to produce strong marginal proposals for both interacting agents
    before a joint decoder ranks/refines their combinations.
    """

    def create_scene_level_data(self, index):
        info_meta = self.infos[index]
        scene_id = info_meta["scenario_id"]
        with open(self.data_path / f"sample_{scene_id}.pkl", "rb") as f:
            info = pickle.load(f)

        sdc_track_index = info["sdc_track_index"]
        current_time_index = info["current_time_index"]
        timestamps = np.asarray(
            info["timestamps_seconds"][: current_time_index + 1],
            dtype=np.float32,
        )

        track_infos = info["track_infos"]
        obj_types_raw = np.asarray(track_infos["object_type"])
        obj_ids_raw = np.asarray(track_infos["object_id"])
        obj_trajs_full = track_infos["trajs"]
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        pair_indices = self._select_pair_indices(
            info, obj_trajs_full, obj_types_raw, current_time_index
        )
        pair_indices = self._canonicalize_pair_indices(
            pair_indices, obj_ids_raw
        )
        center_track_index = pair_indices.astype(np.int64)

        center_objects, center_track_index = self.get_interested_agents(
            track_index_to_predict=center_track_index,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types_raw,
            scene_id=scene_id,
        )
        if len(center_track_index) != 2:
            raise RuntimeError(
                f"Scenario {scene_id} did not keep both interactive agents"
            )

        (
            obj_trajs_data,
            obj_trajs_mask,
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            center_gt_trajs,
            center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new,
            sdc_track_index_new,
            obj_types,
            obj_ids,
        ) = self.create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=center_track_index,
            sdc_track_index=sdc_track_index,
            timestamps=timestamps,
            obj_types=obj_types_raw,
            obj_ids=obj_ids_raw,
        )

        pair_object_ids = obj_ids_raw[center_track_index]
        pair_filtered_indices = []
        for object_id in pair_object_ids:
            matches = np.where(obj_ids == object_id)[0]
            if len(matches) == 0:
                raise RuntimeError(
                    f"Pair object id {object_id} was filtered out in {scene_id}"
                )
            pair_filtered_indices.append(int(matches[0]))
        pair_filtered_indices = np.asarray(pair_filtered_indices, dtype=np.int64)

        # Use the first agent's local frame as the official joint frame.
        pair_gt_trajs = obj_trajs_future_state[0, pair_filtered_indices][None]
        pair_gt_trajs_mask = obj_trajs_future_mask[0, pair_filtered_indices][None]
        pair_gt_trajs[pair_gt_trajs_mask == 0] = 0

        ret_dict = {
            "scenario_id": np.asarray([scene_id]),
            "obj_trajs": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": track_index_to_predict_new,
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "obj_types": obj_types,
            "obj_ids": obj_ids,
            "center_objects_world": center_objects,
            "center_objects_id": pair_object_ids,
            "center_objects_type": obj_types_raw[center_track_index],
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "center_gt_trajs_src": obj_trajs_full[center_track_index],
            "pair_partner_index": np.asarray([[pair_filtered_indices[1], pair_filtered_indices[0]]], dtype=np.int64),
            "pair_track_index_to_predict": center_track_index[None].astype(np.int64),
            "pair_object_ids": pair_object_ids[None].astype(np.int64),
            "pair_object_types": obj_types_raw[center_track_index][None],
            "pair_center_objects_world": obj_trajs_full[
                center_track_index, current_time_index
            ][None].astype(np.float32),
            "pair_frame_world": center_objects[:1][None].astype(np.float32),
            "pair_current_state": self._build_pair_current_state(
                obj_trajs_data, pair_filtered_indices
            ),
            "pair_gt_trajs": pair_gt_trajs.astype(np.float32),
            "pair_gt_trajs_mask": pair_gt_trajs_mask.astype(bool),
            "pair_gt_trajs_src": obj_trajs_full[center_track_index][None].astype(np.float32),
            "pair_is_official_ooi": np.asarray(
                [
                    str(self.dataset_cfg.get("PAIR_SELECTION", "closest")).lower()
                    in {
                        "objects_of_interest",
                        "official_objects_of_interest",
                        "official_ooi",
                    }
                ],
                dtype=bool,
            ),
        }


        traffic_nodes, traffic_mask = self._build_traffic_light_nodes(
            info, center_objects[0]
        )
        ret_dict["traffic_light_nodes"] = traffic_nodes
        ret_dict["traffic_light_mask"] = traffic_mask

        if not self.dataset_cfg.get("WITHOUT_HDMAP", False):
            if info["map_infos"]["all_polylines"].__len__() == 0:
                info["map_infos"]["all_polylines"] = np.zeros((2, 7), dtype=np.float32)
                print(f"Warning: empty HDMap {scene_id}")

            (
                map_polylines_data,
                map_polylines_mask,
                map_polylines_center,
            ) = self.create_map_data_for_center_objects(
                center_objects=center_objects,
                map_infos=info["map_infos"],
                center_offset=self.dataset_cfg.get(
                    "CENTER_OFFSET_OF_MAP", (30.0, 0)
                ),
            )

            ret_dict["map_polylines"] = map_polylines_data
            ret_dict["map_polylines_mask"] = map_polylines_mask > 0
            ret_dict["map_polylines_center"] = map_polylines_center

        return ret_dict


def waymo_interaction_evaluation(
    pred_dicts,
    eval_second=8,
    num_modes_for_eval=6,
    normalize_scores=True,
):
    if eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6
    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10
    elif eval_second == 8:
        num_frames_in_total = 91
        num_frame_to_eval = 16
    else:
        raise ValueError(f"Unsupported eval_second={eval_second}")

    sampled_interval = 5
    num_scenario = len(pred_dicts)
    num_agents = 2
    top_k = min(num_modes_for_eval, pred_dicts[0]["pred_trajs"].shape[0])

    pred_trajs = np.zeros(
        (num_scenario, 1, top_k, num_agents, num_frame_to_eval, 2),
        dtype=np.float32,
    )
    pred_scores = np.zeros((num_scenario, 1, top_k), dtype=np.float32)
    gt_trajs = np.zeros((num_scenario, num_agents, num_frames_in_total, 7), dtype=np.float32)
    gt_is_valid = np.zeros((num_scenario, num_agents, num_frames_in_total), dtype=bool)
    pred_gt_indices = np.zeros((num_scenario, 1, num_agents), dtype=np.int64)
    pred_gt_indices_mask = np.ones((num_scenario, 1, num_agents), dtype=bool)
    object_type = np.zeros((num_scenario, num_agents), dtype=np.int64)
    object_type_cnt = Counter()

    for scene_idx, item in enumerate(pred_dicts):
        scores = np.nan_to_num(
            np.asarray(item["pred_scores"], dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        order = np.argsort(scores)[::-1][:top_k]
        selected_scores = np.clip(scores[order], 0.0, 1.0)
        if normalize_scores:
            selected_scores = selected_scores / np.maximum(
                selected_scores.sum(), 1e-8
            )
        pred_scores[scene_idx, 0, : len(order)] = selected_scores
        pred_trajs[scene_idx, 0, : len(order)] = np.asarray(
            item["pred_trajs"], dtype=np.float32
        )[order, :, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]

        cur_gt = np.asarray(item["gt_trajs"], dtype=np.float32)
        gt_trajs[scene_idx] = cur_gt[:, :num_frames_in_total, :][
            :, :, [0, 1, 3, 4, 6, 7, 8]
        ]
        gt_is_valid[scene_idx] = cur_gt[:, :num_frames_in_total, -1].astype(bool)
        pred_gt_indices[scene_idx, 0] = np.arange(num_agents, dtype=np.int64)
        for agent_idx, type_name in enumerate(item["object_type"]):
            object_type[scene_idx, agent_idx] = object_type_to_id[str(type_name)]
            object_type_cnt[str(type_name)] += 1

    eval_config = _default_metrics_config(
        eval_second=eval_second, num_modes_for_eval=top_k
    )
    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=tf.convert_to_tensor(pred_trajs, tf.float32),
        prediction_score=tf.convert_to_tensor(pred_scores, tf.float32),
        ground_truth_trajectory=tf.convert_to_tensor(gt_trajs, tf.float32),
        ground_truth_is_valid=tf.convert_to_tensor(gt_is_valid, tf.bool),
        prediction_ground_truth_indices=tf.convert_to_tensor(
            pred_gt_indices, tf.int64
        ),
        prediction_ground_truth_indices_mask=tf.convert_to_tensor(
            pred_gt_indices_mask, tf.bool
        ),
        object_type=tf.convert_to_tensor(object_type, tf.int64),
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
    result_dict = {}
    avg_results = {
        f"{metric} - {obj_type}": [0.0, 0]
        for metric in ["minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]
        for obj_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    }
    for metric_idx, metric in enumerate(
        ["minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]
    ):
        for name_idx, breakdown_name in enumerate(metric_names):
            cur_type = breakdown_name.split("_")[1]
            value = float(metric_results[metric_idx][name_idx])
            result_dict[f"{metric} - {breakdown_name}"] = value
            avg_results[f"{metric} - {cur_type}"][0] += value
            avg_results[f"{metric} - {cur_type}"][1] += 1
    for key, (value, count) in avg_results.items():
        result_dict[key] = value / max(count, 1)
    for metric in ["minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]:
        result_dict[metric] = float(
            np.mean(
                [
                    result_dict[f"{metric} - {obj_type}"]
                    for obj_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
                ]
            )
        )
    for key, value in object_type_cnt.items():
        result_dict[f"count/{key}"] = float(value)
    result_dict["scenario_count"] = float(num_scenario)

    result_format_str = (
        f"Waymo Interaction eval_second={eval_second} "
        f"mAPv2={result_dict['mAP']:.6f}, "
        f"minADE={result_dict['minADE']:.6f}, "
        f"minFDE={result_dict['minFDE']:.6f}, "
        f"MissRate={result_dict['MissRate']:.6f}, "
        f"OverlapRate={result_dict['OverlapRate']:.6f}"
    )
    return result_dict, result_format_str
