# LLM in Triton

### 1. Build Model (Hugging Face -> Pytorch -> ONNX -> TensorRT)

[**IMPORTANT**](https://github.com/microsoft/onnxruntime/issues/10905#issuecomment-1072649358): ONNX optimization in not applicable on TensorRT compilation, prefer original ONNX 


```bash
# Build Env
docker build -t llm_trt_exporter .
docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash

# Hugging Face -> Pytorch -> ONNX
python3 transform.py

# ONNX -> TensorRT
trtexec --onnx=opt_125m/model.onnx \
        --saveEngine=opt_125m/model.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:8x512,attention_mask:8x512 \
        --maxShapes=input_ids:32x512,attention_mask:32x512 \
        --memPoolSize=workspace:2048 \
        --fp16
mv opt_125m/model.plan models/opt_125m_model/1/ # for TensorRT
mv opt_125m/config.json models/opt_125m_tokenizer/1/ # for Tokenizer
mv opt_125m/tokenizer.json models/opt_125m_tokenizer/1/ # for Tokenizer
```



<details><summary> Export ONNX : optimu-cli (NOT Recommend !!)</summary>

```bash
# always raise bus error...
optimum-cli export onnx --framework pt \
                        --optimize=O1 \
                        --task text-generation \
                        --atol 1e-5 \
                        --model facebook/opt-125m opt-125m/
```
</details>


<details><summary> Export ONNX : transformers.onnx (NOT Recommend !!)</summary>


```bash
# Deprecated soon, Remove in v5, but still work w/o error
python -m transformers.onnx --model=facebook/opt-6.7b out2/
```
</details>

---

### 2. Setup Triton IS
```bash
docker-compose build 
docker-compose up
```

### 3. Client

```bash
docker exec -it controller bash
# testing / benchmark
perf_analyzer -m opt_125m_tokenizer -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1  --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape text:1
perf_analyzer -m opt_125m_model -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --shape input_ids:128 --shape attention_mask:128
perf_analyzer -m opt_125m -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape TEXT:1

# HTTP Inference client
python3 send_request.py -u triton:8000 --batch_size 1 --n_iter 4 --statistics -m opt_125m_tokenizer -i text -o input_ids:attention_mask
python3 send_request.py -u triton:8000 --batch_size 1 --n_iter 4 --statistics -m opt_125m -i TEXT -o LOGITS
```

---

### TODO

[] [fastertransformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

[] gRPC inference client