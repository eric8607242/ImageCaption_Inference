from .config import PATH_TO_EXTRACT_CONFIG_FILE, PATH_TO_VOCAB_STOI, PATH_TO_VOCAB_ITOS
from .extract_feature import inititalize_extrace_model, extract_features
from .dlct import initialize_caption_model, inference_caption_model

def inference(model_states, path_to_image):
    features = extract_features(model_states["extract_model"], path_to_image)
    caption_text = inference_caption_model(
        model_states["caption_model"],
        features,
        model_states["text_field"],
    )
    return caption_text

def initialize_model_states(
        path_to_extract_model_weight, 
        path_to_caption_model_weight,
    ):
    extract_model = inititalize_extrace_model(
        PATH_TO_EXTRACT_CONFIG_FILE,
        path_to_extract_model_weight
    )

    caption_model, text_field = initialize_caption_model(
        path_to_caption_model_weight,
        PATH_TO_VOCAB_STOI,
        PATH_TO_VOCAB_ITOS,
    )

    return {
        "extract_model": extract_model,
        "caption_model": caption_model,
        "text_field": text_field
    }
    
