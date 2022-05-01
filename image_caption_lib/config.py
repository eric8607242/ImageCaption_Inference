import os
path_to_current_dir = os.path.dirname(__file__)

PATH_TO_EXTRACT_CONFIG_FILE = f"{path_to_current_dir}/configs/config.yaml"
PATH_TO_VOCAB_STOI = f"{path_to_current_dir}/configs/vocab_stoi.json"
PATH_TO_VOCAB_ITOS = f"{path_to_current_dir}/configs/vocab_itos.json"

import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
