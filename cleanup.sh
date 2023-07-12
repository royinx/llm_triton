# clean up Triton model repository

find $1 -iname "*model.plan*"     -exec rm -rf {} \;
find $1 -iname "*model.onnx*"     -exec rm -rf {} \;
find $1 -iname "*.model*"         -exec rm -rf {} \;
find $1 -iname "*.json*"          -exec rm -rf {} \;
find $1 -iname "*__pycache__*"    -exec rm -rf {} \;

# find . -iname "*.onnx" -not -path "*TensorRT/*" -exec rm -rf {} \;
