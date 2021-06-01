# YolactEdge: Real-time Instance Segmentation on the Edge
```
██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗    ███████╗██████╗  ██████╗ ███████╗
╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
 ╚████╔╝ ██║   ██║██║     ███████║██║        ██║       █████╗  ██║  ██║██║  ███╗█████╗  
  ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║       ██╔══╝  ██║  ██║██║   ██║██╔══╝  
   ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║       ███████╗██████╔╝╚██████╔╝███████╗
   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

**YolactEdge**, the first competitive instance segmentation approach that runs on small edge devices at real-time speeds. Specifically, YolactEdge runs at up to 30.8 FPS on a Jetson AGX Xavier (and 172.7 FPS on an RTX 2080 Ti) with a ResNet-101 backbone on 550x550 resolution images. This is the code for [our paper](https://arxiv.org/abs/2012.12259).

**For a real-time demo and more samples, check out our [demo video](https://www.youtube.com/watch?v=GBCK9SrcCLM).**

[![example-gif-1](data/yolact_edge_example_1.gif)](https://www.youtube.com/watch?v=GBCK9SrcCLM)

[![example-gif-2](data/yolact_edge_example_2.gif)](https://www.youtube.com/watch?v=GBCK9SrcCLM)

[![example-gif-3](data/yolact_edge_example_3.gif)](https://www.youtube.com/watch?v=GBCK9SrcCLM)

## Model Zoo

We provide baseline YOLACT and YolactEdge models trained on COCO and YouTube VIS (our sub-training split, with COCO joint training).

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands.

YouTube VIS models:

| Method | Backbone&nbsp; | mAP | AGX-Xavier FPS | RTX 2080 Ti FPS | weights |
|:-------------:|:-------------:|:----:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|
| YOLACT | R-50-FPN | 44.7 | 8.5 | 59.8 | [download](https://drive.google.com/file/d/1EfoQ0OteuQdY2yU9Od8XHTHrizQVFR2w/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHLweuem6riY6lVK?e=cUGBRf) |
| YolactEdge <br>(w/o TRT) | R-50-FPN | 44.2| 10.5 | 67.0 | [download](https://drive.google.com/file/d/1qvd4W28yzzXFb2wwGfYySv5HHzGU26XP/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHGgB-KrQLubo7eZ?e=h26XJM) |
| YolactEdge | R-50-FPN | 44.0| 32.4 | 177.6 | [download](https://drive.google.com/file/d/1qvd4W28yzzXFb2wwGfYySv5HHzGU26XP/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHGgB-KrQLubo7eZ?e=h26XJM) |
| YOLACT | R-101-FPN | 47.3 | 5.9 | 42.6 | [download](https://drive.google.com/file/d/1doS5MRhpSs4puVCuzR5i3GrDMSxcw7Lx/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHOei4kogT1JCfO7?e=dLcrVg) |
| YolactEdge <br>(w/o TRT) | R-101-FPN | 46.9| 9.5 | 61.2 | [download](https://drive.google.com/file/d/1mSxesVaMmYc13cPHiEnRvubPxy8WBjJW/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHAqrmvsL1RMH9WK?e=Tnlu7p) |
| YolactEdge | R-101-FPN | 46.2 | 30.8 | 172.7 | [download](https://drive.google.com/file/d/1mSxesVaMmYc13cPHiEnRvubPxy8WBjJW/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiHAqrmvsL1RMH9WK?e=Tnlu7p) |

COCO models:

| Method | &nbsp;&nbsp;&nbsp;Backbone&nbsp;&nbsp;&nbsp;&nbsp; | mAP | Titan Xp FPS | AGX-Xavier FPS | RTX 2080 Ti FPS | weights |
|:-------------:|:-------------:|:----:|:----:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|
| YOLACT | MobileNet-V2 | 22.1 | - | 15.0 | 35.7 | [download](https://drive.google.com/file/d/1L4N4VcykqE-D5JUgWW9zBd6WKmZPBAZQ/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=RraXLv) |
| YolactEdge | MobileNet-V2 | 20.8 | - | 35.7 | 161.4 | [download](https://drive.google.com/file/d/1L4N4VcykqE-D5JUgWW9zBd6WKmZPBAZQ/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=RraXLv) |
| YOLACT | R-50-FPN | 28.2 | 42.5 | 9.1 | 45.0 | [download](https://drive.google.com/file/d/15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG5ZnhPTSkqBCURo?e=lNOaXr) |
| YolactEdge | R-50-FPN | 27.0| - | 30.7 | 140.3 | [download](https://drive.google.com/file/d/15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG5ZnhPTSkqBCURo?e=lNOaXr) |
| YOLACT | R-101-FPN | 29.8 | 33.5 | 6.6 | 36.5 | [download](https://drive.google.com/file/d/1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=HyfH8Z) |
| YolactEdge | R-101-FPN | 29.5 | - | 27.3 | 124.8 | [download](https://drive.google.com/file/d/1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp/view?usp=sharing) \| [mirror](https://1drv.ms/u/s!AkSxI62eEcpbiG8nFXtvgAkI-c1H?e=HyfH8Z) |

## Installation

See [INSTALL.md](INSTALL.md).

Optionally, you can use the official [Dockerfile](docker) to set up full enivronment with one command.

## Getting Started

Follow the [installation instructions](INSTALL.md) to set up required environment for running YolactEdge.

See instructions to [evaluate](https://github.com/haotian-liu/yolact_edge#evaluation) and [train](https://github.com/haotian-liu/yolact_edge#training) with YolactEdge.

### Colab Notebook

Try out our [Colab Notebook](https://colab.research.google.com/drive/1Mzst4q4Y-SQszIHhlEv1CkT4hwja4GNw?usp=sharing) with a live demo to learn about basic usage.

If you are interested in evaluating YolactEdge with TensorRT, we provide another [Colab Notebook](https://colab.research.google.com/drive/1nEZAYnGbF7VetqltAlUTyAGTI71MvPPF?usp=sharing) with TensorRT environment configuration on Colab.

## Evaluation

### Quantitative Results
```Shell
# Convert each component of the trained model to TensorRT using the optimal settings and evaluate on the YouTube VIS validation set (our split).
python3 eval.py --trained_model=./weights/yolact_edge_vid_847_50000.pth

# Evaluate on the entire COCO validation set.
python3 eval.py --trained_model=./weights/yolact_edge_54_800000.pth

# Output a COCO JSON file for the COCO test-dev. The command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively. These files can then be submitted to the website for evaluation.
python3 eval.py --trained_model=./weights/yolact_edge_54_800000.pth --dataset=coco2017_testdev_dataset --output_coco_json
```

### Qualitative Results
```Shell
# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.3.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --display
```

### Benchmarking

```Shell
# Benchmark the trained model on the COCO validation set.
# Run just the raw model on the first 1k images of the validation set
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --benchmark --max_images=1000
```

### Notes

#### Handling inference error when using TensorRT
If you are using TensorRT conversion of YolactEdge and encountered issue in PostProcessing or NMS stage, this might be related to TensorRT engine issues. We implemented a experimental safe mode that will handle these cases carefully. Try this out with `--use_tensorrt_safe_mode` option in your command.


#### Inference using models trained with YOLACT
If you have a pre-trained model with [YOLACT](https://github.com/dbolya/yolact), and you want to take advantage of either TensorRT feature of YolactEdge, simply specify the `--config=yolact_edge_config` in command line options, and the code will automatically detect and convert the model weights to be compatible.

```Shell
python3 eval.py --config=yolact_edge_config --trained_model=./weights/yolact_base_54_800000.pth
```


#### Inference without Calibration

If you want to run inference command without calibration, you can either run with FP16-only TensorRT optimization, or without TensorRT optimization with corresponding configs. Refer to `data/config.py` for examples of such configs.

```Shell
# Evaluate YolactEdge with FP16-only TensorRT optimization with '--use_fp16_tensorrt' option (replace all INT8 optimization with FP16).
python3 eval.py --use_fp16_tensorrt --trained_model=./weights/yolact_edge_54_800000.pth

# Evaluate YolactEdge without TensorRT optimization with '--disable_tensorrt' option.
python3 eval.py --disable_tensorrt --trained_model=./weights/yolact_edge_54_800000.pth
```

### Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --images=path/to/input/folder:path/to/output/folder
```
### Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If video_multiframe > 1, then the trt_batch_size should be increased to match it or surpass it. 
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --video_multiframe=2 --trt_batch_size 2 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --video_multiframe=2 --trt_batch_size 2 --video=0

# Process a video and save it to another file. This is unoptimized.
python eval.py --trained_model=weights/yolact_edge_54_800000.pth --score_threshold=0.3 --top_k=100 --video=input_video.mp4:output_video.mp4
```
Use the help option to see a description of all available command line arguments:
```Shell
python eval.py --help
```
### Programmatic inference

You can use yolact_edge as a package in your own code. There are two steps to make this work:
 1) Install YOLACT edge as python package: ```pip install .```
 2) Use it as in the example provided in ```pkg_usage.py```

## Training
Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
   - For MobileNetV2, download `mobilenet_v2-b0353104.pth` from [here](https://drive.google.com/file/d/1F8YAAWITIkZ_w-fVeetmQKMkfGYfHvUM/view?usp=sharing).
 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base edge config with a batch size of 8 (the default).
python train.py --config=yolact_edge_config

# Resume training yolact_edge with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_edge_config --resume=weights/yolact_edge_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

### Training on video dataset
```Shell
# Pre-train the image based model
python train.py --config=yolact_edge_youtubevis_config

# Train the flow (warping) module
python train.py --config=yolact_edge_vid_trainflow_config --resume=./weights/yolact_edge_youtubevis_847_50000.pth

# Fine tune the network jointly
python train.py --config=yolact_edge_vid_config --resume=./weights/yolact_edge_vid_trainflow_144_100000.pth
```


### Custom Datasets
You can also train on your own dataset by following these steps:
 - Depending on the type of your dataset, create a COCO-style (image) or YTVIS-style (video) Object Detection JSON annotation file for your dataset. The specification for this can be found here for [COCO](http://cocodataset.org/#format-data) and [YTVIS](https://github.com/youtubevos/cocoapi) respectively. Note that we don't use some fields, so the following may be omitted:
   - `info`
   - `liscense`
   - Under `image`: `license, flickr_url, coco_url, date_captured`
   - `categories` (we use our own format for categories, see below)
 - Create a definition for your dataset under `dataset_base` in `data/config.py` (see the comments in `dataset_base` for an explanation of each field):
```Python
my_custom_dataset = dataset_base.copy({
    'name': 'My Dataset',

    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',

    'has_gt': True,
    'class_names': ('my_class_id_1', 'my_class_id_2', 'my_class_id_3', ...),

    # below is only needed for YTVIS-style video dataset.

    # whether samples all frames or key frames only.
    'use_all_frames': False,

    # the following four lines define the frame sampling strategy for the given dataset.
    'frame_offset_lb': 1,
    'frame_offset_ub': 4,
    'frame_offset_multiplier': 1,
    'all_frame_direction': 'allway',

    # 1 of K frames is annotated
    'images_per_video': 5,

    # declares a video dataset
    'is_video': True
})
```
 - Note that: class IDs in the annotation file should start at 1 and increase sequentially on the order of `class_names`. If this isn't the case for your annotation file (like in COCO), see the field `label_map` in `dataset_base`.
 - Finally, in `yolact_edge_config` in the same file, change the value for `'dataset'` to `'my_custom_dataset'` or whatever you named the config object above and `'num_classes'` to number of classes in your dataset+1. Then you can use any of the training commands in the previous section.
 

## Citation

If you use this code base in your work, please consider citing:

```
@inproceedings{yolactedge-icra2021,
  author    = {Haotian Liu and Rafael A. Rivera Soto and Fanyi Xiao and Yong Jae Lee},
  title     = {YolactEdge: Real-time Instance Segmentation on the Edge},
  booktitle = {ICRA},
  year      = {2021},
}
```
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

## Contact
For questions about our paper or code, please contact [Haotian Liu](mailto:lhtliu@ucdavis.edu) or [Rafael A. Rivera-Soto](mailto:riverasoto@ucdavis.edu).
