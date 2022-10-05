#!/bin/bash

SOURCE_CODE=$1
DATASETS=$2

docker build -t yolact_edge:11.4_cuda8.2 -f Dockerfile .

docker run --gpus all -it --name=yolact_edge \
  --shm-size=64gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $SOURCE_CODE:/root/yolact_edge/:rw \
  -v $DATASETS:/datasets/:ro \
  yolact_edge:11.4_cuda8.2
