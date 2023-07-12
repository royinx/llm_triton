import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import T5Tokenizer, TensorType # , AutoTokenizer
class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        # path: str = os.path.join(args["model_repository"], args["model_version"], "config")
        path: str = os.path.join(args["model_repository"], args["model_version"], "config")
        self.tokenizer = T5Tokenizer.from_pretrained(path, use_fast=True)
        # self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", use_fast=True)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [t[0].decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]
            tokens: Dict[str, np.ndarray] = self.tokenizer(text=query,
                                                           return_tensors=TensorType.NUMPY)
            # tensorrt uses int32 as input type, ort uses int64
            tokens = tokens["input_ids"].astype(np.int32)
            # communicate the tokenization results to Triton server
            tensor_input = pb_utils.Tensor("input_ids", tokens)
            inference_response = pb_utils.InferenceResponse(output_tensors=[tensor_input])
            responses.append(inference_response)

        return responses