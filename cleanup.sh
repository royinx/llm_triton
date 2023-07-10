# clean up Triton model repository
# export TARGET_DIR=model_repository
export TARGET_DIR=models

find $TARGET_DIR -iname "*model.plan*"     -exec rm -rf {} \;
find $TARGET_DIR -iname "*config.json*"    -exec rm -rf {} \;
find $TARGET_DIR -iname "*tokenizer.json*" -exec rm -rf {} \;
find $TARGET_DIR -iname "*__pycache__*"    -exec rm -rf {} \;

# find . -iname "*.onnx" -not -path "*TensorRT/*" -exec rm -rf {} \;