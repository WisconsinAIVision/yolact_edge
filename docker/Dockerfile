FROM nvcr.io/nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y  \
    git wget build-essential \
    python3 python3-setuptools python3-pip python3-dev python3-tk \
    ffmpeg libsm6 libxext6
RUN ln -svf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade --force pip

# CUDNN
ARG version="8.2.2.26-1+cuda11.4"
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages \
    libcudnn8=${version} libcudnn8-dev=${version} && apt-mark hold libcudnn8 libcudnn8-dev

# Install Tensorrt 8.2.1.8
ARG version="8.2.1-1+cuda11.4"
RUN apt-get update && \
    apt-get install -y libnvinfer8=${version} libnvonnxparsers8=${version} libnvparsers8=${version} libnvinfer-plugin8=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python3-libnvinfer=${version} && \
    apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer


# # Install dependencies
RUN pip install cython opencv-python pillow matplotlib GitPython termcolor tensorboard packaging
RUN pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# torch2trt_dynamic
WORKDIR /root
RUN git clone https://github.com/grimoire/torch2trt_dynamic.git torch2trt_dynamic && \
    cd torch2trt_dynamic && \
    python setup.py develop

# installing plugins for torch2trt_dynamic
WORKDIR /root

RUN apt install -y software-properties-common && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt update && apt install -y cmake && \
    git clone --depth=1 --branch v0.5.0 https://github.com/grimoire/amirstan_plugin.git && \
    cd amirstan_plugin && \
    cmake -DTENSORRT_DIR=/usr/lib/x86_64-linux-gnu -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && \
    make -j$(nproc)

ENV AMIRSTAN_LIBRARY_PATH=/root/amirstan_plugin/lib

WORKDIR /root/yolact_edge

ENV LANG C.UTF-8
