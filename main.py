import time
import os
import json

from image_caption_lib import initialize_model_states, inference

if __name__ == "__main__":
    path_to_extract_model_weight = "X-101.pth"
    path_to_caption_model_weight = "pretrained_model.pth"
    path_to_image = "image.jpeg"

    model_states = initialize_model_states(
        path_to_extract_model_weight,
        path_to_caption_model_weight,
    )

    caption_text = inference(model_states, path_to_image)
    print(caption_text)
