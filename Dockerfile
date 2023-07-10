FROM nvcr.io/nvidia/tensorrt:23.06-py3
RUN pip3 install nvidia-pyindex
COPY requirements.txt .
RUN pip3 install -r requirements.txt