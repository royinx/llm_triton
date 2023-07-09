# use SequenceClassification for this example
from optimum.onnxruntime import ORTModelForSequenceClassification as Model

from transformers import AutoTokenizer


def export(model_checkpoint, save_directory):
    # Load a model from transformers and export it to ONNX
    ort_model = Model.from_pretrained(model_checkpoint, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Save the onnx model and tokenizer
    ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # for advanced model exporting , check optimum.exporters.onnx.main_export

if __name__ == "__main__":
    # Decoder models
    model_chkpts = ["facebook/opt-125m",
                    "facebook/opt-350m",
                    "facebook/opt-1.3b",
                    "facebook/opt-6.7b"]

    model_checkpoint = model_chkpts[0]
    save_directory = "opt_125m/"
    export(model_checkpoint, save_directory)

    # gpt_neox 
    # t5
    # mt5
    # longt5