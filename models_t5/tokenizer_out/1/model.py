import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import T5Tokenizer, AutoTokenizer

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x,axis=axis, keepdims=True)

class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        path: str = os.path.join(args["model_repository"], args["model_version"], "config")
        self.tokenizer = T5Tokenizer.from_pretrained(path, use_fast=True)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast=True)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = [] # 
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "decoder_hidden_states").as_numpy()
            decoder_outputs = softmax(logits).argmax(axis=-1)
            semantic_outputs = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens = True)
            
            if isinstance(semantic_outputs, list):
                semantic_outputs = " ".join(semantic_outputs).strip()

            np_obj = np.array([str(x).encode('utf-8') for x in semantic_outputs], dtype=np.object_) 
            tensor_input = [pb_utils.Tensor("output", np_obj)]

            inference_response = pb_utils.InferenceResponse(output_tensors=tensor_input)
            
            responses.append(inference_response) # append single response per request

        return responses