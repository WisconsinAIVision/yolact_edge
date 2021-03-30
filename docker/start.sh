#!/bin/bash

SOURCE=$1
DATASETS=$2

docker run --rm \
	   -ti \
           --name yolact_edge \
           --gpus all \
           -v $SOURCE:/home/yolact_edge/yolact_edge/:rw \
           -v $DATASETS:/datasets/:ro \
           yolact_edge

./into.sh
