"""JFER motion decoder."""

import torch

from ..jfer import JFERCore
from models.decoder.base_decoder import BaseMotionDecoder


class JFERDecoder(BaseMotionDecoder):
    """Add structured joint exploration to the base motion decoder.

    Base scene encoding, object/map attention, dense prediction, and motion
    heads remain implemented by :class:`BaseMotionDecoder`. This subclass owns
    the joint candidate initialization, per-layer joint coupling, residual
    refinement, guarded final selection, and their training loss.
    """

    _SUPPORTED_TYPES = {"jfer"}

    def __init__(self, in_channels, config):
        super().__init__(in_channels=in_channels, config=config)
        self.integrated_joint_world_cfg = self.model_cfg.get(
            "INTEGRATED_JOINT_WORLD", None
        )
        if (
            self.integrated_joint_world_cfg is None
            or not self.integrated_joint_world_cfg.get("ENABLED", False)
        ):
            raise ValueError("JFER requires INTEGRATED_JOINT_WORLD.ENABLED=True")

        world_type = str(
            self.integrated_joint_world_cfg.get("TYPE", "")
        ).lower()
        if world_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                "JFER only packages the JFER candidate-bank decoder, got: "
                + world_type
            )

        map_dim = self.model_cfg.get("MAP_D_MODEL", self.d_model)
        self.integrated_joint_world = JFERCore(
            cfg=self.integrated_joint_world_cfg,
            query_dim=self.d_model,
            map_dim=map_dim,
            num_future_frames=self.num_future_frames,
            num_decoder_layers=self.num_decoder_layers,
        )

    def apply_transformer_decoder(
        self,
        center_objects_feature,
        center_objects_type,
        obj_feature,
        obj_mask,
        obj_pos,
        dense_future_trajs,
        map_feature,
        map_mask,
        map_pos,
        input_dict,
    ):
        intention_query, intention_points = self.get_motion_query(
            center_objects_type
        )
        world_state = self.integrated_joint_world.prepare_worlds(
            center_feature=center_objects_feature,
            obj_feature=obj_feature,
            obj_mask=obj_mask,
            map_feature=map_feature,
            map_mask=map_mask,
            input_dict=input_dict,
            obj_pos=obj_pos,
            map_pos=map_pos,
        )
        (
            intention_query,
            intention_points,
            query_content,
            world_state,
        ) = self.integrated_joint_world.initialize_queries(
            intention_query=intention_query,
            intention_points=intention_points,
            state=world_state,
        )
        self.forward_ret_dict["intention_points"] = intention_points.permute(
            1, 0, 2
        )

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]
        center_objects_feature = center_objects_feature[None].repeat(
            num_query, 1, 1
        )

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        dynamic_query_center = intention_points
        pred_list = []
        world_outputs = []

        for layer_idx in range(self.num_decoder_layers):
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature,
                kv_mask=obj_mask,
                kv_pos=obj_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
            )

            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos,
                map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=(
                    self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES
                ),
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query,
            )
            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature,
                kv_mask=map_mask,
                kv_pos=map_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.map_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=True,
                query_index_pair=collected_idxs,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps,
            )

            query_feature = torch.cat(
                [
                    center_objects_feature,
                    obj_query_feature,
                    map_query_feature,
                ],
                dim=-1,
            )
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1)
            query_content, world_state = (
                self.integrated_joint_world.couple_queries(
                    layer_idx=layer_idx,
                    query_content=query_content,
                    state=world_state,
                )
            )

            flat_query = query_content.permute(1, 0, 2).contiguous().view(
                num_center_objects * num_query, -1
            )
            pred_scores = self.motion_cls_heads[layer_idx](flat_query).view(
                num_center_objects, num_query
            )
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](flat_query).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    5,
                )
                pred_vel = self.motion_vel_heads[layer_idx](flat_query).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    2,
                )
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](flat_query).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    7,
                )

            world_output = self.integrated_joint_world.condition_predictions(
                layer_idx=layer_idx,
                query_content=query_content,
                pred_scores=pred_scores,
                pred_trajs=pred_trajs,
                state=world_state,
                input_dict=input_dict,
            )
            query_content = world_output["query_content"]
            pred_scores = world_output["agent_pred_scores"]
            pred_trajs = world_output["agent_pred_trajs"]
            if layer_idx + 1 < self.num_decoder_layers:
                intention_query, intention_points = (
                    self.integrated_joint_world.reorder_query_anchors(
                        intention_query=intention_query,
                        intention_points=intention_points,
                        order=world_output["mode_order"],
                    )
                )
            world_outputs.append(world_output)
            pred_list.append([pred_scores, pred_trajs])

            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[
                :, :, -1, 0:2
            ].contiguous().permute(1, 0, 2)

        self.forward_ret_dict["query_content"] = query_content.permute(
            1, 0, 2
        ).contiguous()
        if self.integrated_joint_world.consistent_mode_reordering:
            self.forward_ret_dict["intention_points"] = (
                intention_points.permute(1, 0, 2)
            )
        self.forward_ret_dict["integrated_joint_world_state"] = world_state
        self.forward_ret_dict["integrated_joint_world_rets"] = world_outputs
        assert len(pred_list) == self.num_decoder_layers
        return pred_list

    def get_loss(self, tb_pre_tag=""):
        direct_joint_modes = bool(
            self.integrated_joint_world_cfg.get("DIRECT_SPARSE_MODES", False)
        )
        decoder_weight = float(
            self.integrated_joint_world_cfg.get(
                "BASE_DECODER_LOSS_WEIGHT", 0.0
            )
            if direct_joint_modes
            else 1.0
        )
        dense_weight = float(
            self.integrated_joint_world_cfg.get(
                "DENSE_FUTURE_LOSS_WEIGHT", 0.2
            )
            if direct_joint_modes
            else 1.0
        )

        tb_dict = {}
        disp_dict = {}
        total_loss = self.forward_ret_dict["pred_list"][-1][0].new_zeros(())
        if decoder_weight > 0.0:
            decoder_loss, decoder_tb, decoder_disp = self.get_decoder_loss(
                tb_pre_tag=tb_pre_tag
            )
            total_loss = total_loss + decoder_weight * decoder_loss
            tb_dict.update(decoder_tb)
            disp_dict.update(decoder_disp)
        if dense_weight > 0.0:
            dense_loss, tb_dict, disp_dict = (
                self.get_dense_future_prediction_loss(
                    tb_pre_tag=tb_pre_tag,
                    tb_dict=tb_dict,
                    disp_dict=disp_dict,
                )
            )
            total_loss = total_loss + dense_weight * dense_loss

        world_loss, world_tb = self.integrated_joint_world.get_loss(
            self.forward_ret_dict["integrated_joint_world_state"],
            self.forward_ret_dict["integrated_joint_world_rets"],
            self.forward_ret_dict["input_dict"],
        )
        world_scale = float(
            self.integrated_joint_world_cfg.get("LOSS_SCALE", 1.0)
        )
        total_loss = total_loss + world_scale * world_loss
        for key, value in world_tb.items():
            tb_dict[f"{tb_pre_tag}{key}"] = value
        disp_dict[f"{tb_pre_tag}loss_joint_world"] = (
            world_scale * world_tb["loss_integrated_joint_world"]
        )
        if "joint_oracle_ade" in world_tb:
            disp_dict[f"{tb_pre_tag}joint_ade"] = world_tb[
                "joint_oracle_ade"
            ]
        for key in (
            "loss_candidate_residual_flow",
            "loss_residual_geometry",
            "loss_residual_score",
            "joint_oracle_fde",
            "candidate_residual_abs",
        ):
            if key in world_tb:
                disp_dict[f"{tb_pre_tag}{key}"] = world_tb[key]

        tb_dict[f"{tb_pre_tag}loss"] = total_loss.item()
        disp_dict[f"{tb_pre_tag}loss"] = total_loss.item()
        return total_loss, tb_dict, disp_dict

    def forward(self, batch_dict):
        input_dict = batch_dict["input_dict"]
        for key in (
            "cur_epoch",
            "cur_iter",
            "total_epochs",
            "total_iters_each_epoch",
            "accumulated_iter",
        ):
            if key in batch_dict:
                input_dict[key] = batch_dict[key]

        obj_feature = batch_dict["obj_feature"]
        obj_mask = batch_dict["obj_mask"]
        obj_pos = batch_dict["obj_pos"]
        map_feature = batch_dict["map_feature"]
        map_mask = batch_dict["map_mask"]
        map_pos = batch_dict["map_pos"]
        center_feature = batch_dict["center_objects_feature"]
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        center_feature = self.in_proj_center_obj(center_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        projected_obj = obj_feature.new_zeros(
            num_center_objects, num_objects, obj_feature_valid.shape[-1]
        )
        projected_obj[obj_mask] = obj_feature_valid
        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        projected_map = map_feature.new_zeros(
            num_center_objects, num_polylines, map_feature_valid.shape[-1]
        )
        projected_map[map_mask] = map_feature_valid

        projected_obj, dense_trajs = self.apply_dense_future_prediction(
            obj_feature=projected_obj,
            obj_mask=obj_mask,
            obj_pos=obj_pos,
        )
        self.forward_ret_dict["decoder_obj_feature"] = projected_obj
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_feature,
            center_objects_type=input_dict["center_objects_type"],
            obj_feature=projected_obj,
            obj_mask=obj_mask,
            obj_pos=obj_pos,
            dense_future_trajs=dense_trajs,
            map_feature=projected_map,
            map_mask=map_mask,
            map_pos=map_pos,
            input_dict=input_dict,
        )
        self.forward_ret_dict["pred_list"] = pred_list
        self.forward_ret_dict["input_dict"] = input_dict

        if self.training:
            self.forward_ret_dict["center_gt_trajs"] = input_dict[
                "center_gt_trajs"
            ]
            self.forward_ret_dict["center_gt_trajs_mask"] = input_dict[
                "center_gt_trajs_mask"
            ]
            self.forward_ret_dict["center_gt_final_valid_idx"] = input_dict[
                "center_gt_final_valid_idx"
            ]
            self.forward_ret_dict["obj_trajs_future_state"] = input_dict[
                "obj_trajs_future_state"
            ]
            self.forward_ret_dict["obj_trajs_future_mask"] = input_dict[
                "obj_trajs_future_mask"
            ]
            self.forward_ret_dict["center_objects_type"] = input_dict[
                "center_objects_type"
            ]
            return batch_dict

        final_output = self.forward_ret_dict[
            "integrated_joint_world_rets"
        ][-1]
        pred_scores, pred_trajs = self.integrated_joint_world.select_final(
            final_output
        )
        batch_dict["pred_scores"] = pred_scores
        batch_dict["pred_trajs"] = pred_trajs
        return batch_dict
