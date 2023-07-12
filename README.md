# LLM in Triton

[**IMPORTANT**](https://github.com/microsoft/onnxruntime/issues/10905#issuecomment-1072649358): ONNX optimization in not applicable on TensorRT compilation, prefer original ONNX

Platform: 3950x + 32GB + 2080ti (11GB)

### 0. Build Env
```bash
# Create Triton IS
docker-compose build
docker build -t llm_trt_exporter .
docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash
```

---

### 1. Build Model (Hugging Face -> Pytorch -> ONNX -> TensorRT)

Choose `ONLY ONE` of the following model and execute

#### 1.1. Hugging Face -> Pytorch -> ONNX

<details open><summary> I. Sequence Classifier ( optimum )</summary>

```bash
export SRC_DIR=model_zoo/opt_125m_SeqCls
export LLMTASK=seqcls   # { seqcls , qa , encode }
python3 transform.py SeqCls
```

</details>


<details><summary> II. encode ( transformers.onnx )</summary>

```bash
export SRC_DIR=model_zoo/squad2_tran_onnx/
export LLMTASK=encode   # { seqcls , qa , encode }
python -m transformers.onnx --model=deepset/roberta-base-squad2 $SRC_DIR
```

</details>

<details><summary> III. QA ( optimu-cli )</summary>

```bash
export SRC_DIR=model_zoo/squad2_qa/
export LLMTASK=qa   # { seqcls , qa , encode }
optimum-cli export onnx --framework pt \
                        --task question-answering \
                        --atol 1e-5 \
                        --model deepset/roberta-base-squad2 $SRC_DIR
```
</details>

<details><summary> IV. T5 ( NNDF )</summary>

Check [details](docs.t5.md)

</details>

<!-- <details><summary> :x: 5. CausalLM ( torch.onnx.export )</summary>

```bash
python3 transform.py CausalLM
```
</details> -->



#### 1.2 .ONNX -> TensorRT
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
# copy config.pbtxt
cp config/ensemble/${LLMTASK}_config.pbtxt      models/llm/config.pbtxt
cp config/model/${LLMTASK}_config.pbtxt         models/model/config.pbtxt
cp config/tokenizer/config.pbtxt                models/tokenizer/config.pbtxt

# Copy Model to Triton
mv $SRC_DIR/model.plan         models/model/1/          # Copy model to TRITON
mv $SRC_DIR/*                  models/tokenizer/1/      # Copy Tokenizer to TRITON

# Create Triton IS
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
#                    QA - START_LOGITS:END_LOGITS
#                encode - HIDDEN_STATES
```

</details>

---

### 4. Clean up
```bash
# clean up Triton
sh cleanup.sh
sudo sh unittest.sh
```


### TODO
- [x] T5

- [ ] General torch.onnx.export for CausalLM (In Progress)

- [ ] New Precision released - [Efficient 4-bit Inference (NF4, FP4)](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.40.0) & [FP4 bitsandbytes colab notebook](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

- [ ] logits to text ( QA )

- [ ] [fastertransformer_backend](https://github.com/triton-inference-server/fastertransformer_backend)

- [ ] gRPC inference client
