import os
from typing import Dict, List

import torch
from torch.nn.functional import softmax as softmax_torch
import numpy as np
from transformers import T5Tokenizer , AutoTokenizer
path: str = os.path.join("config")
# tokenizer = T5Tokenizer.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast=True)
# request = np.ones((1, 128, 32128), dtype=np.float32)  # (4, 128, 32128)
logits = np.random.rand(4, 128, 32128)   # (4, 128, 32128)
# logits = np.random.rand(1, 10, 32128)   # (4, 128, 32128)


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x,axis=axis, keepdims=True)


decoder_outputs = softmax(logits).argmax(axis=-1)
print(decoder_outputs.shape)
semantic_outputs = tokenizer.batch_decode(decoder_outputs, skip_special_tokens = True)
data = np.array([str(x).encode('utf-8') for x in semantic_outputs], dtype=np.object_)
print(data.shape)

# for decoder_output in decoder_outputs:
#     semantic_outputs = tokenizer.decode(decoder_output, skip_special_tokens = True)

#     if isinstance(semantic_outputs, list):
#         semantic_outputs = " ".join(semantic_outputs).strip()
#     responses.append(semantic_outputs)

# encoded = np.array([str(x).encode('utf-8') for x in responses], dtype=np.object_)
# print(encoded.shape)