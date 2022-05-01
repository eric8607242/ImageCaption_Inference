from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.config import CfgNode as CN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from torchvision.io import read_image
from torchvision import transforms

from .grid_feats import add_attribute_config

def wrap_image_to_inputs(model, path_to_image):
    image = read_image(path_to_image)
    w, h = image.shape[1:]

    down_scale = transforms.Compose([transforms.Scale((640, int(h/(w/640))))])
    image = down_scale(image)
    inputs = [{"image": image}]

    inputs = model.preprocess_image(inputs)
    return inputs


def inititalize_extrace_model(
    path_to_extract_config_file,
    path_to_extract_model_weight,
    opts=[]
):
    cfg = setup_cfg(path_to_extract_config_file, 
                    path_to_extract_model_weight,
                    opts)

    model = build_model(cfg)
    DetectionCheckpointer(
        model, 
        save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    return model

def setup_cfg(path_to_extract_config_file,
              path_to_extract_model_weight,
              opts):

    cfg = get_cfg()
    add_attribute_config(cfg)

    cfg.merge_from_file(path_to_extract_config_file)
    cfg.merge_from_list(opts)

    cfg.MODEL.WEIGHTS = path_to_extract_model_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    #default_setup(cfg, args)
    return cfg
