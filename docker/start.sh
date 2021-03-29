#!/bin/bash

DATASETS=$1

docker run --rm \
	   -ti \
           --name yolact_edge \
           --gpus all \
           -v ..:/home/yolact_edge/yolact_edge/:rw \
           -v $DATASETS:/datasets/:ro \
           yolact_edge

./into.sh
