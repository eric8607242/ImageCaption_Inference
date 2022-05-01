from detectron2.evaluation import inference_context
import torch
import torch.nn.functional as F

def extract_grid_feature(model, inputs):
    with torch.no_grad():
        with inference_context(model):
            features = model.backbone(inputs.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            outputs = F.adaptive_avg_pool2d(outputs, (7, 7))
            outputs = torch.squeeze(outputs)
            outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = outputs.transpose(0, 1)

            grid_features = {
                "grids": outputs.cpu().numpy()
            }
    return grid_features


