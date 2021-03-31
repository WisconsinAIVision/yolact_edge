#!/bin/bash

SOURCE=$1
DATASETS=$2
#for example from yolact_edge folder ./docker/start.sh `pwd` <path_to_datasets>

docker run --rm \
	   -ti \
           --name yolact_edge \
           --gpus all \
           -v $SOURCE:/home/docker/yolact_edge/:rw \
           -v $DATASETS:/datasets/:ro \
           yolact_edge_image
