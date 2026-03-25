import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from ..utils import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
        layer_ensemble (int): Number of final decoder layers to average
            logits over. 0 or 1 = use only last layer (default behavior).
    """
    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 layer_ensemble=0):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.layer_ensemble = layer_ensemble

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes."""
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='trunc')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds)   
        final_scores = scores 
        final_preds = labels 

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            limit = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= limit[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= limit[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!'
            )

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes with optional decoder layer ensemble."""
        all_cls = preds_dicts['all_cls_scores']  # [nb_dec, bs, num_query, cls]
        all_box = preds_dicts['all_bbox_preds']  # [nb_dec, bs, num_query, 10]

        if self.layer_ensemble > 1:
            # Average logits over last N decoder layers, use last layer's boxes
            n = min(self.layer_ensemble, all_cls.shape[0])
            cls_scores = all_cls[-n:].mean(dim=0)  # [bs, num_query, cls]
        else:
            cls_scores = all_cls[-1]

        bbox_preds = all_box[-1]  # always use last layer's boxes

        batch_size = cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(cls_scores[i], bbox_preds[i]))

        return predictions_list
