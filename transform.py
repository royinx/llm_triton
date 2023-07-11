# built-in library
import sys
sys.path.insert(0, "plugin")
import os

# third party
import torch
from transformers import OPTForCausalLM, OPTForQuestionAnswering # model only
from transformers import PretrainedConfig # config.json only

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import onnx

def export_t5(model_checkpoint, save_directory) -> None:

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).eval()
    # print(isinstance(model,torch.nn.Module))

    # setup input
    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate sequence for an input
    outputs = model.generate(inputs.input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))



    inputs = tokenizer("translate English to German: That is good.", return_tensors="pt")
    print(tokenizer.decode(model.generate(inputs["input_ids"])[0],
                           skip_special_tokens=True))


    from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
    from T5.T5ModelConfig import T5ModelTRTConfig, T5Metadata
    from NNDF.networks import NetworkMetadata, Precision
    metadata=NetworkMetadata(variant=model_checkpoint,
                             precision=Precision(fp16=True),
                             other=T5Metadata(kv_cache=False))

    if not os.path.exists(os.path.join( save_directory , "encoder")): os.makedirs(os.path.join( save_directory, "encoder"))
    if not os.path.exists(os.path.join( save_directory , "decoder")): os.makedirs(os.path.join( save_directory, "decoder"))
    encoder_onnx_model_fpath = os.path.join( save_directory , "encoder", "encoder.onnx")
    decoder_onnx_model_fpath = os.path.join( save_directory , "decoder", "decoder-with-lm-head.onnx")

    t5_encoder = T5EncoderTorchFile(model, metadata)
    t5_decoder = T5DecoderTorchFile(model, metadata)

    onnx_t5_encoder = t5_encoder.as_onnx_model(encoder_onnx_model_fpath, force_overwrite=False)
    print(encoder_onnx_model_fpath)

    onnx_t5_decoder = t5_decoder.as_onnx_model(decoder_onnx_model_fpath, force_overwrite=False)
    print(decoder_onnx_model_fpath)
    print("Done!")

def export_CausalLM(model_checkpoint, save_directory) -> None:
    r"""
    export model in lower level , using torch.onnx.export
    """
    # generate tokenizer
    # AutoModelForQuestionAnswering
    # model = OPTForCausalLM.from_pretrained(model_checkpoint).eval()

    # model = OPTForQuestionAnswering.from_pretrained(model_checkpoint).eval()
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.save_pretrained(save_directory)

    # generate config.json
    pretrain_cfg = PretrainedConfig.from_pretrained(model_checkpoint, export=True)
    output_config_file = os.path.join(save_directory, "config.json")
    pretrain_cfg.to_json_file(output_config_file, use_diff=True)

    # generate model.onnx
    prompt = "Hey, are you consciours? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # print(type(model))
    # x = model(inputs["input_ids"],inputs["attention_mask"])
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, export=True)
    model.save_pretrained(save_directory)

    with torch.no_grad():
        symbolic_names = {0: "batch_size", 1: "sequence_length"}
        torch.onnx.export(
            model,
            args=(inputs["input_ids"],inputs["attention_mask"]),
            f=f"{save_directory}/model.onnx",
            export_params=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"input_ids": symbolic_names, "attention_mask": symbolic_names, "output": symbolic_names},
            do_constant_folding=True,
            opset_version=15,  # Use a suitable opset version, such as 12 or 13
        )
    onnx_model = onnx.load(f"{save_directory}/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Done!")

def export_SeqCls(model_checkpoint, save_directory):
    r"""
    using higher level API to export model
    """
    # Load a model from transformers and export it to ONNX
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
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

    model_checkpoint, save_directory = model_chkpts[0] , "opt_125m"

    model_type = sys.argv[1]
    if model_type == "SeqCls":
        export_SeqCls(model_checkpoint,     os.path.join("model_zoo", f"{save_directory}_SeqCls")) # model_zoo/opt_125m_SeqCls
    elif model_type == "t5":
        model_checkpoint = "google/flan-t5-small"
        # model_checkpoint = "google/flan-t5-xl"
        export_t5(model_checkpoint,         os.path.join("model_zoo",f"t5"))                       # model_zoo/t5
    else:
        raise NotImplementedError