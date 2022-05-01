# DLCT_ImageCaption_Inference

This repository is an unofficial **model inference** codebase for the papr Dual-Level Collaborative Transformer for Image Captioning (CVPR2021).

In this repo, we incorporate the repo [image-captioning-DLCT](https://github.com/luo3300612/image-captioning-DLCT) and [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa) to make image caption model inference more easily. 

***Given the image path, yYou can get the image caption sentence with only two function calls (`initialize_model_states` and `inference`)!***

The steps of this inference repo are show as follows:
1. grid features extractions from the repo [original_script](https://github.com/facebookresearch/grid-feats-vqa)
2. region features extractions from [original_script](https://github.com/luo3300612/image-captioning-DLCT/blob/main/others/extract_region_feature.py)
3. the alignment graphs extraction from [original_script](https://github.com/luo3300612/image-captioning-DLCT/blob/main/align/align.ipynb)
4. image caption model inference with all features above [original_script](https://github.com/luo3300612/image-captioning-DLCT/blob/main/eval.py)

# Requirements
- Download feature extraction [pretrained model X-101](https://dl.fbaipublicfiles.com/grid-feats-vqa/X-101/X-101.pth) from [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
- Download image caption [pretrained model](https://pan.baidu.com/s/1xVZO7t8k4H_l3aEyuA-KXQ) (acess code: jcj6) from [image-captioning-DLCT](https://github.com/luo3300612/image-captioning-DLCT).
- Install python dependency
```
pip install -r requirements.txt
```
- Install `en_core_web_sm`
```
python3 -m spacy download en_core_web_sm
```
- Install Detectron 2 following [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
```

# Get Start
## Inference
```
from image_caption_lib import initialize_model_states, inference

path_to_extract_model_weight = "..."
path_to_caption_model_weight = "..."

path_to_inference_image = "..."


model_states = initialize_model_states(
    path_to_extract_model_weight,
    path_to_caption_model_weight,
)

caption_text = inference(model_states, path_to_image)
```

# Acknowledgements
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
- [image-captioning-DLCT](https://github.com/luo3300612/image-captioning-DLCT)
