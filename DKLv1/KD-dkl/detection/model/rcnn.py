# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from mdistiller.distillers.DKD import dkd_loss

from .teacher import build_teacher
from .reviewkd import build_kd_trans, hcl


__all__ = ["RCNNKD", "ProposalNetwork"]

def rcnn_dkd_loss(stu_predictions, tea_predictions, gt_classes, alpha, beta, temperature):
    stu_logits, stu_bbox_offsets = stu_predictions
    tea_logits, tea_bbox_offsets = tea_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)
    loss_dkd = dkd_loss(stu_logits, tea_logits, gt_classes, alpha, beta, temperature)
    return {
        'loss_dkd': loss_dkd,
    }

@META_ARCH_REGISTRY.register()
class RCNNKD(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        teacher_pixel_mean: Tuple[float],
        teacher_pixel_std: Tuple[float],
        teacher: nn.Module,
        kd_args,
        input_format: Optional[str] = None,
        teacher_input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.teacher = teacher
        self.kd_args = kd_args
        if self.kd_args.TYPE in ("ReviewKD", "ReviewDKD"):
            self.kd_trans = build_kd_trans(self.kd_args)

        self.input_format = input_format
        self.teacher_input_format = teacher_input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("teacher_pixel_mean", torch.tensor(teacher_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("teacher_pixel_std", torch.tensor(teacher_pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "kd_args": cfg.KD,
            "teacher": build_teacher(cfg),
            "teacher_input_format": cfg.TEACHER.INPUT.FORMAT,
            "teacher_pixel_mean": cfg.TEACHER.MODEL.PIXEL_MEAN,
            "teacher_pixel_std": cfg.TEACHER.MODEL.PIXEL_STD,
            
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        losses = {}
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.kd_args.TYPE == "DKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            stu_predictions = self.forward_pure_roi_head(self.roi_heads, features, sampled_proposals)
            tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads, t_features, sampled_proposals)
            detector_losses.update(rcnn_dkd_loss(
                stu_predictions, tea_predictions, [x.gt_classes for x in sampled_proposals], 
                self.kd_args.DKD.ALPHA, self.kd_args.DKD.BETA, self.kd_args.DKD.T))
        elif self.kd_args.TYPE == "ReviewKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            t_features = [t_features[f] for f in t_features]
            s_features = [features[f] for f in features]
            s_features = self.kd_trans(s_features)
            losses['loss_reviewkd'] = hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
        elif self.kd_args.TYPE == "ReviewDKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            # dkd loss
            stu_predictions = self.forward_pure_roi_head(self.roi_heads, features, sampled_proposals)
            tea_predictions = self.forward_pure_roi_head(self.teacher.roi_heads, t_features, sampled_proposals)
            detector_losses.update(rcnn_dkd_loss(
                stu_predictions, tea_predictions, [x.gt_classes for x in sampled_proposals], 
                self.kd_args.DKD.ALPHA, self.kd_args.DKD.BETA, self.kd_args.DKD.T))
            # reviewkd loss
            t_features = [t_features[f] for f in t_features]
            s_features = [features[f] for f in features]
            s_features = self.kd_trans(s_features)
            losses['loss_reviewkd'] = hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
        else:
            raise NotImplementedError(self.kd_args.TYPE)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return RCNNKD._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def teacher_preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.teacher_pixel_mean) / self.teacher_pixel_std for x in images]
        if self.input_format != self.teacher_input_format:
            images = [x.index_select(0,torch.LongTensor([2,1,0]).to(self.device)) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
