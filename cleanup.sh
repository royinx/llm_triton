# clean up Triton model repository
# export TARGET_DIR=model_repository
export TARGET_DIR=models_t5

find $TARGET_DIR -iname "*model.plan*"     -exec rm -rf {} \;
find $TARGET_DIR -iname "*model.onnx*"     -exec rm -rf {} \;
find $TARGET_DIR -iname "*.model*"         -exec rm -rf {} \;
find $TARGET_DIR -iname "*.json*"          -exec rm -rf {} \;
find $TARGET_DIR -iname "*__pycache__*"    -exec rm -rf {} \;

# find . -iname "*.onnx" -not -path "*TensorRT/*" -exec rm -rf {} \;
