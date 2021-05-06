FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

#installation settings
ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade --user"
ARG GIT_CLONE="git clone --depth 10"

#user settings
ARG USER=docker
ARG UID=1000
ARG GID=1000
ARG PW=user

# Add user and his password
RUN useradd -m ${USER} --uid=${UID} \
    && echo "${USER}:${PW}" | chpasswd \
    && usermod -s /bin/bash ${USER}

WORKDIR /home/${USER}

#tools
RUN apt-get update && $APT_INSTALL \
    nano \
    git \
    python3 \
    python3-setuptools \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 && \
    pip3 install --upgrade pip 
    
RUN $PIP_INSTALL \
    cython \
    opencv-python \
    pillow \
    matplotlib \
    GitPython \
    termcolor \
    tensorboard

RUN $PIP_INSTALL git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"

#pytorch  
RUN $PIP_INSTALL torch==1.7.1+cu110 \
                 torchvision==0.8.2+cu110 \
                 -f https://download.pytorch.org/whl/torch_stable.html

#install TensorRT packages
ENV version="8.0.5.39-1+cuda11.0"
RUN apt-get update && $APT_INSTALL \
                    libcudnn8=${version} \
                    libcudnn8-dev=${version} && \
                    apt-mark hold libcudnn8 libcudnn8-dev

ENV version="7.2.3-1+cuda11.0"
RUN apt-get update && $APT_INSTALL \
                libnvinfer7=${version} \ 
                libnvonnxparsers7=${version} \
                libnvparsers7=${version} \
                libnvinfer-plugin7=${version} \
                libnvinfer-dev=${version} \
                libnvonnxparsers-dev=${version} \
                libnvparsers-dev=${version} \
                libnvinfer-plugin-dev=${version} \
                python-libnvinfer=${version} \
                python3-libnvinfer=${version} && \
                apt-mark hold libnvinfer7 \
                libnvonnxparsers7 \
                libnvparsers7 \
                libnvinfer-plugin7 \
                libnvinfer-dev \ 
                libnvonnxparsers-dev \
                libnvparsers-dev \
                libnvinfer-plugin-dev \
                python-libnvinfer \
                python3-libnvinfer

#clone and compile torch2trt
RUN $GIT_CLONE https://github.com/NVIDIA-AI-IOT/torch2trt

WORKDIR /home/${USER}/torch2trt
RUN python3 setup.py install --plugins

#cleaning
RUN apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

#start path
WORKDIR /home/${USER}/yolact_edge
