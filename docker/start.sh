#!/bin/bash

SOURCE_CODE=$1
DATASETS=$2

docker run --gpus all -it --name=yolact_edge \
   --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -p 8082:22 -p 8083:6006 \
  -v $SOURCE_CODE:/yolact_edge/:rw \
  -v $DATASETS:/datasets/:rw \
  yolact_edge_image
