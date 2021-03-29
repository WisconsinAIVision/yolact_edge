#!/bin/bash

docker run --rm \
	   -ti \
           -n yolact_edge \
           --gpus all \
           -v ..:/home/yolact_edge/yolact_edge/:rw
           yolact_edge

./into.sh
