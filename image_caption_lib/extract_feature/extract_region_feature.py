import numpy as np

from detectron2.evaluation import inference_context
import torch

def extract_region_feature(model, inputs):
    thresh = 0.5
    max_regions = 100
    with torch.no_grad():
        with inference_context(model):
            features = model.backbone(inputs.tensor)

            proposals, _ = model.proposal_generator(inputs, features)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            features = [features[f] for f in model.roi_heads.in_features]
            box_features1 = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = model.roi_heads.box_head(box_features1)

            predictions = model.roi_heads.box_predictor(box_features)
            pred_instances, index = model.roi_heads.box_predictor.inference(predictions, proposals)

            topk = 10
            scores = pred_instances[0].get_fields()['scores']
            topk_index = index[0][:topk]

            thresh_mask = scores > thresh
            thresh_index = index[0][thresh_mask]

            if len(thresh_index) < 10:
                index = [topk_index]
            elif len(thresh_index) > max_regions:
                index = [thresh_index[:max_regions]]
            else:
                index = [thresh_index]


            proposal_box_features1 = box_features1[index].mean(dim=[2,3])
            proposal_box_features = box_features[index]
            boxes = pred_instances[0].get_fields()['pred_boxes'].tensor[:len(index[0])]

            image_size = pred_instances[0].image_size

            assert boxes.shape[0] == proposal_box_features.shape[0]

            region_features = {
                "features": proposal_box_features1.cpu().numpy(),
                "boxes": boxes.cpu().numpy(),
                "size": np.array([image_size])
            }

    return region_features



