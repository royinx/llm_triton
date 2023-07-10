# !/bin/bash

# build all modles
docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash -c "python3 transform.py SeqCls"
docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash -c "python -m transformers.onnx --model=deepset/roberta-base-squad2 model_zoo/squad2_tran_onnx/"
# set +e # ignore error from optimum-cli
docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter bash -c "optimum-cli export onnx --framework pt --task question-answering --atol 1e-5 --model deepset/roberta-base-squad2 model_zoo/squad2_qa/"
# set -e

sudo chown -R $USER ./*

check (){
        docker-compose down
        docker run --rm -it -v $PWD:/py -w /py --runtime nvidia llm_trt_exporter \
                trtexec --onnx=$1/model.onnx \
                        --saveEngine=$1/model.plan \
                        --minShapes=input_ids:1x1,attention_mask:1x1 \
                        --optShapes=input_ids:8x512,attention_mask:8x512 \
                        --maxShapes=input_ids:32x512,attention_mask:32x512 \
                        --memPoolSize=workspace:2048 \
                        --fp16

        # copy config.pbtxt
        cp config/ensemble/${2}_config.pbtxt      models/llm/config.pbtxt
        cp config/model/${2}_config.pbtxt         models/model/config.pbtxt
        cp config/tokenizer/config.pbtxt                models/tokenizer/config.pbtxt

        # Copy Model to Triton
        cp $1/model.plan         models/model/1/          # Copy model to TRITON
        cp $1/config.json        models/tokenizer/1/      # Copy Tokenizer to TRITON
        cp $1/tokenizer.json     models/tokenizer/1/      # Copy Tokenizer to TRITON

        # Create Triton IS
        docker-compose up -d

        sleep 10 # Wait for Triton to load the model

}

check model_zoo/opt_125m_SeqCls seqcls
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 -m llm -i TEXT -o LOGITS" > log && echo "" >> log
sh cleanup.sh

check model_zoo/squad2_tran_onnx encode
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 --batch_size 8 -m llm -i TEXT -o HIDDEN_STATE" >> log  && echo "" >> log
sh cleanup.sh

check model_zoo/squad2_qa qa
docker exec -it controller sh -c "python3 send_request.py -u triton:8000 --batch_size 8 -m llm -i TEXT -o START_LOGITS:END_LOGITS" >> log  && echo "" >> log
sh cleanup.sh

docker-compose down
# # clean up model_zoo
find model_zoo -iname "*onnx*" -exec rm -rf {} \;
rm -rf model_zoo/*