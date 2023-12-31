# LLM in Triton

[**IMPORTANT**](https://github.com/microsoft/onnxruntime/issues/10905#issuecomment-1072649358): ONNX optimization in not applicable on TensorRT compilation, prefer original ONNX

Platform: 3950x + 32GB + 2080ti (11GB)

### 0. Build Env
```bash
# access trt_builder
docker build -t llm_trt_exporter .
docker run --rm -it --name trt_builder -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash
```

---

### 1. Build Model (Hugging Face -> Pytorch -> ONNX -> TensorRT)

Choose `ONLY ONE` of the following model and execute

#### Hugging Face -> Pytorch -> ONNX

<details open><summary> A. Sequence Classifier ( optimum )</summary>

```bash
export SRC_DIR=model_zoo/opt_125m_SeqCls
export LLMTASK=seqcls   # { seqcls , qa , encode }
python3 transform.py SeqCls
```

</details>


<details><summary> B. encode ( transformers.onnx )</summary>

```bash
export SRC_DIR=model_zoo/squad2_tran_onnx/
export LLMTASK=encode   # { seqcls , qa , encode }
python -m transformers.onnx --model=deepset/roberta-base-squad2 $SRC_DIR
```

</details>

<details><summary> C. QA ( optimu-cli )</summary>

```bash
export SRC_DIR=model_zoo/squad2_qa/
export LLMTASK=qa   # { seqcls , qa , encode }
optimum-cli export onnx --framework pt \
                        --task question-answering \
                        --atol 1e-5 \
                        --model deepset/roberta-base-squad2 $SRC_DIR
```
</details>

<details><summary> D. T5 ( NNDF )</summary>

Check [details](docs/t5.md)

</details>

<details><summary> E. CausalLM ( torch.onnx.export )</summary>

> :x: Still in progress
```bash
python3 transform.py CausalLM
```
</details>

---

#### ONNX -> TensorRT
```bash
trtexec --onnx=$SRC_DIR/model.onnx \
        --saveEngine=$SRC_DIR/model.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:8x512,attention_mask:8x512 \
        --maxShapes=input_ids:32x512,attention_mask:32x512 \
        --memPoolSize=workspace:2048 \
        --fp16
```

---

### 2. Setup Triton Inference Server

```bash
export DST_DIR=triton-server/models
# copy config.pbtxt
cp config/ensemble/${LLMTASK}_config.pbtxt      $DST_DIR/llm/config.pbtxt
cp config/model/${LLMTASK}_config.pbtxt         $DST_DIR/model/config.pbtxt
cp config/tokenizer/config.pbtxt                $DST_DIR/tokenizer/config.pbtxt

# Copy Model to Triton
mv $SRC_DIR/model.plan         $DST_DIR/model/1/          # Copy model to TRITON
mv $SRC_DIR/*                  $DST_DIR/tokenizer/1/      # Copy Tokenizer to TRITON

# Exit trt_builder
```

```bash
# Create Triton IS
docker-compose build
docker-compose up
```

---

### 3. Client

#### Quick test
```bash
# SeqCls
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 -m llm -i TEXT -o LOGITS --statistics"
# Encode
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 --batch_size 8 -m llm -i TEXT -o HIDDEN_STATE"
# QA
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 --batch_size 8 -m llm -i TEXT -o START_LOGITS:END_LOGITS"
```
<details><summary> Manual test </summary>

####
```bash
docker exec -it controller bash
# testing / benchmark
perf_analyzer -m tokenizer -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1  --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape text:1  # tokenizer
perf_analyzer -m model -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --shape input_ids:128 --shape attention_mask:128 # model
perf_analyzer -m llm -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape TEXT:1  # ensemble

# HTTP Inference client
python3 send_request.py -u triton:8000 -m tokenizer -i text -o input_ids:attention_mask --statistics # Tokenizer
python3 send_request.py -u triton:8000 -m llm -i TEXT -o LOGITS --statistics  # Ensemble
# output layer = seqcls - LOGITS
#                encode - HIDDEN_STATE
#                    QA - START_LOGITS:END_LOGITS
```

</details>

---

### 4. Clean up
```bash
# clean up Triton
sh cleanup.sh triton-server/models
sudo sh unittest.sh
```


<!--
---

<details><summary> :white_check_mark: 2. QA ( transformers.onnx )</summary>

### 2.1. Build Model (Hugging Face -> Pytorch -> ONNX -> TensorRT)

```bash
# Hugging Face -> Pytorch -> ONNX
python -m transformers.onnx --model=deepset/roberta-base-squad2 model_zoo/squad2_tran_onnx/

# ONNX -> TensorRT
trtexec --onnx=model_zoo/squad2_tran_onnx/model.onnx \
        --saveEngine=model_zoo/squad2_tran_onnx/model.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:4x512,attention_mask:4x512 \
        --maxShapes=input_ids:8x512,attention_mask:8x512 \
        --memPoolSize=workspace:2048 \
        --fp16

# Copy model to TRITON
mv model_zoo/squad2_tran_onnx/model.plan          model_repository/models_qa/squad2_qa_model/1/

# Copy Tokenizer to TRITON
mv model_zoo/squad2_tran_onnx/config.json         model_repository/models_qa/squad2_qa_tokenizer/1/
mv model_zoo/squad2_tran_onnx/tokenizer.json      model_repository/models_qa/squad2_qa_tokenizer/1/
```

---

### 2.2. Setup Triton IS
```bash
# make sure last docker-compose is down  and changes to new volume
docker-compose up
```

### 2.3. Client

```bash
docker exec -it controller bash
# testing / benchmark
perf_analyzer -m squad2_qa_tokenizer -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1  --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape text:1
perf_analyzer -m squad2_qa_model -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --shape input_ids:128 --shape attention_mask:128
perf_analyzer -m squad2_qa -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape TEXT:1

# HTTP Inference client
python3 send_request.py -u triton:8000 -m squad2_qa_tokenizer -i text -o input_ids:attention_mask --statistics # Tokenizer
python3 send_request.py -u triton:8000 -m squad2_qa -i TEXT -o LOGITS --statistics  # Ensemble
```

---
</details>


<details><summary> :heavy_check_mark: 3. QA ( optimu-cli )</summary>


### 3.1. Build Model


```bash
# Hugging Face -> Pytorch -> ONNX
optimum-cli export onnx --framework pt \
                        --task question-answering \
                        --atol 1e-5 \
                        --model deepset/roberta-base-squad2 model_zoo/squad2_QA/

# ONNX -> TensorRT
trtexec --onnx=model_zoo/squad2_opt/model.onnx \
        --saveEngine=model_zoo/squad2_opt//model.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:4x512,attention_mask:4x512 \
        --maxShapes=input_ids:8x512,attention_mask:8x512 \
        --memPoolSize=workspace:2048 \
        --fp16

# Copy model to TRITON
mv model_zoo/squad2_opt//model.plan         model_repository/models_qa/squad2_qa_model/1/

# Copy Tokenizer to TRITON
mv model_zoo/squad2_opt//config.json        model_repository/models_qa/squad2_qa_tokenizer/1/
mv model_zoo/squad2_opt//tokenizer.json     model_repository/models_qa/squad2_qa_tokenizer/1/
```

---

### 3.2. Setup Triton IS
```bash
# make sure last docker-compose is down  and changes to new volume
docker-compose up
```

### 3.3. Client

```bash
docker exec -it controller bash
# testing / benchmark
perf_analyzer -m squad2_qa_tokenizer -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1  --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape text:1
perf_analyzer -m squad2_qa_model -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --shape input_ids:128 --shape attention_mask:128
perf_analyzer -m squad2_qa -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape TEXT:1

# HTTP Inference client
python3 send_request.py -u triton:8000 --batch_size 8 -m squad2_qa_tokenizer -i text -o input_ids:attention_mask --statistics # Tokenizer
python3 send_request.py -u triton:8000 --batch_size 8 -m squad2_qa -i TEXT -o START_LOGITS:END_LOGITS  # Ensemble
```

---
</details>





<details><summary> :white_check_mark: 4. QA ( torch.onnx.export )</summary>

### 4.1. Build Model

```bash
# Hugging Face -> Pytorch -> ONNX
python3 transform.py CausalLM

# ONNX -> TensorRT
trtexec --onnx=model_zoo/squad2_torch/model.onnx \
        --saveEngine=model_zoo/squad2_torch/model.plan \
        --minShapes=input_ids:1x1,attention_mask:1x1 \
        --optShapes=input_ids:4x512,attention_mask:4x512 \
        --maxShapes=input_ids:8x512,attention_mask:8x512 \
        --memPoolSize=workspace:2048 \
        --fp16

# Copy model to TRITON
mv model_zoo/squad2_torch/model.plan            model_repository/opt_125m_model/1/

# Copy Tokenizer to TRITON
mv model_zoo/squad2_torch/config.json           model_repository/opt_125m_tokenizer/1/
mv model_zoo/squad2_torch/tokenizer.json        model_repository/opt_125m_tokenizer/1/
```

---
```bash
# always raise bus error...

```

### 4.2. Setup Triton IS
```bash
# make sure last docker-compose is down  and changes to new volume
docker-compose up
```

### 4.3. Client

```bash
docker exec -it controller bash
# testing / benchmark
perf_analyzer -m opt_125m_tokenizer -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1  --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape text:1
perf_analyzer -m opt_125m_model -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --shape input_ids:128 --shape attention_mask:128
perf_analyzer -m opt_125m -u triton:8000 -i HTTP -v -p3000 -d -l3000 -t1 -c5 -b1 --string-data "Hello, I'm Machine Learning Engineer, my duty is " --shape TEXT:1

# HTTP Inference client
python3 send_request.py -u triton:8000 -m opt_125m_tokenizer -i text -o input_ids:attention_mask --statistics # Tokenizer
python3 send_request.py -u triton:8000 -m opt_125m -i TEXT -o LOGITS --statistics  # Ensemble
```

---
</details> -->

<details open><summary> Roadmap </summary>

- [ ] *in Progess* - T5

- [ ] *in Progess* - General torch.onnx.export for CausalLM (In Progress)

- [ ] *Token Requested* - Llama 2

- [ ] Llama 2 with FP4 (Possible on OPT but not TensorRT yet)

- [ ] Derived >> Implement OPT backend for Triton

- [ ] New Precision released - [Efficient 4-bit Inference (NF4, FP4)](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.40.0) & [FP4 bitsandbytes colab notebook](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

- [ ] logits to text ( QA )

- [ ] [fastertransformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

- [ ] gRPC inference client

</details>