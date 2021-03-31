FROM nvcr.io/nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

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

#tensorrt
#torch2trt

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
RUN $PIP_INSTALL torch==1.6.0 \
                 torchvision==0.7.0

#cleaning
RUN apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
