docker build -t yolact_edge -f Dockerfile.xavier  .
docker run -it --rm --net=host --privileged \
           --runtime nvidia -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix/:/tmp/.X11-unix \
           -v $PWD/../:/yolact_edge/:rw \
           --device /dev/video0:/dev/video0 \
           yolact_edge \
           python3 eval.py --trained_model=./weights/yolact_edge_resnet50_54_800000.pth \
                           --score_threshold=0.3 \
                           --top_k=100 \
                           --video_multiframe=2 \
                           --trt_batch_size 2 \
                           --video=0 \
                           --calib_images ./data/coco/images