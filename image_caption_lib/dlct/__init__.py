import multiprocessing
import pickle

import numpy as np
import torch

from ..config import DEVICE

from .data import TextField
from .evaluation import PTBTokenizer
from .model import TransformerEncoder, TransformerDecoderLayer, Transformer, ScaledDotProductAttention

def inference_caption_model(caption_model, features, text_field):
    caption_model.eval()

    region_features = features["region_features"]["features"]
    boxes = features["region_features"]["boxes"]
    size = features["region_features"]["size"]
    grid_features = features["grid_features"]["grids"]
    align_graphs = features["align_graphs"]["mask"]

    size = size[:, ::-1]
    size = np.concatenate([size, size], axis=1)
    relative_boxes = boxes / size

    delta = 100 - region_features.shape[0]
    if delta > 0:
        region_features = np.concatenate([region_features, np.zeros((delta, region_features.shape[1]))], axis=0)
        relative_boxes = np.concatenate([relative_boxes, np.zeros((delta, relative_boxes.shape[1]))], axis=0)
        align_graphs = np.concatenate([align_graphs , np.zeros((delta, align_graphs.shape[1]))], axis=0)
    elif delta < 0:
        region_features = region_features[:100]
        relative_boxes = relative_boxes[:100]
        align_graphs = align_graphs[:100]

    detections = region_features.astype(np.float32)
    boxes = relative_boxes.astype(np.float32)
    grid_features = grid_features.astype(np.float32)
    align_graphs = align_graphs.astype(np.float32)
    detections = torch.tensor(detections)
    detections = detections.unsqueeze(0)

    boxes = torch.tensor(boxes)
    boxes = boxes.unsqueeze(0)

    grid_features= torch.tensor(grid_features)
    grid_features = grid_features.unsqueeze(0)

    align_graphs = torch.tensor(align_graphs)
    align_graphs = align_graphs.unsqueeze(0)

    detections = detections.to(DEVICE)
    boxes = boxes.to(DEVICE)
    grid_features = grid_features.to(DEVICE)
    align_graphs = align_graphs.to(DEVICE)

    with torch.no_grad():
        out, _ = caption_model.beam_search(detections, 20, text_field.stoi['<eos>'], 5, out_size=1,**{'boxes': boxes, 'grids': grid_features, 'masks':align_graphs})
    caps_gen = text_field.decode(out, join_words=False)
    caps_gen1 = text_field.decode(out)

    return caps_gen1

def initialize_caption_model(
        path_to_caption_model_weight,
        path_to_vocab_stoi,
        path_to_vocab_itos,
    ):
    text_field = TextField(
        path_to_vocab_stoi, path_to_vocab_itos,
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )

    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                 d_in=2048,
                                 d_k=64,
                                 d_v=64,
                                 h=8
                                 )
    decoder = TransformerDecoderLayer(len(text_field.stoi), 54, 3, text_field.stoi['<pad>'],
                                      d_k=64,
                                      d_v=64,
                                      h=8
                                      )
    
    model = Transformer(
        text_field.stoi['<bos>'], 
        encoder, decoder, 
        True, True
    ).to(DEVICE)

    data = torch.load(path_to_caption_model_weight)
    model.load_state_dict(data['state_dict'])
    return model, text_field
