## Installation
 - Set up a Python3 environment.
 - Install [Pytorch](http://pytorch.org/) 1.6.0 and TorchVision.
 - Install [TensorRT](https://developer.nvidia.com/tensorrt) 7.1.3.4 and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) 0.1.0 (*optional* for evaluating models without TensorRT, currently TensorRT optimization only supports devices with [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/), and already included in [JetPack SDK](https://developer.nvidia.com/embedded/Jetpack) if using Jetson devices):
   1. Install CUDA 10.2/11.0 and cuDNN 8.0.0.
   2. Download TensorRT 7.1.3.4 tar file [here](https://developer.nvidia.com/nvidia-tensorrt-7x-download) and install TensorRT (refer to [official documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html#installing-tar) for more details).
   ```Shell
   tar xzvf TensorRT-${version}.${os}.${arch}-gnu.${cuda}.${cudnn}.tar.gz
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

   cd TensorRT-${version}/python
   pip3 install tensorrt-*-cp3x-none-linux_x86_64.whl
   
   cd TensorRT-${version}/uff
   pip3 install uff-0.6.9-py2.py3-none-any.whl

   cd TensorRT-${version}/graphsurgeon
   pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl
   ```
   3. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt).
   ```Shell
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   cd torch2trt
   sudo python setup.py install --plugins
   ```

 - Install some other packages:
   ```Shell
   # Cython needs to be installed before pycocotools
   pip install cython
   pip install opencv-python pillow matplotlib
   pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
   pip install GitPython termcolor tensorboard
   ```
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/haotian-liu/yolact_edge.git
   cd yolact_edge
   ```
 - If you'd like to train YolactEdge on COCO, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YolactEdge on COCO `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```
 - To evaluate YolactEdge with TensorRT INT8 calibration you need to download the calibration dataset (this avoids having to download the entire COCO/YouTube-VIS dataset and their annotations) for [COCO](https://drive.google.com/file/d/15jyd5CRJxNiA41UMjGbaSnmaytfeILfI/view?usp=sharing) and [YouTube VIS](https://drive.google.com/file/d/1KT79KHUECdV0fIkBc5OTSHCf13FXg-aO/view?usp=sharing). Store the `calib_images` folder under its corresponding dataset folder as shown in the example below. Note that our best models use INT8 calibration so this step is highly advised.
 - If you'd like to train YolactEdge on YouTube VIS, download the [YouTube VIS dataset](https://youtube-vos.org/dataset/) (you need to register to download) and our training/validation split [annotations](https://drive.google.com/drive/folders/1hFM-BLlsufO-C99QIDSBkD2JR5qVMfx2?usp=sharing) into `./data/YoutubeVIS`.
   - If you'd like to train jointly with COCO, download the COCO dataset and the 2014/2017 annotations using the script above.
   - If you'd like to train on all video frames, download the [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) into `./data/FlyingChairs`.
   - Your dataset folder should be organized like this:
    ```
    ./data
    ├── coco
    │   ├── annotations
    │   ├── calib_images
    │   └── images
    ├── FlyingChairs
    │   ├── data
    │   └── train_val.txt
    └── YoutubeVIS
        ├── annotations
        ├── calib_images
        ├── JPEGImages
        └── train_all_frames
            └── JPEGImages
    ```
