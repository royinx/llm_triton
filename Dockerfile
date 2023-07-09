FROM nvcr.io/nvidia/tensorrt:23.06-py3
COPY requirements.txt .
RUN pip3 install -r requirements.txt