## Installation
 - Set up a Python3 environment.
 - Install [Pytorch](http://pytorch.org/) 1.7.1 and TorchVision v.0.8.2.
 - Install [TensorRT](https://developer.nvidia.com/tensorrt) 8.2.1.8 and [torch2trt_dynamic](https://github.com/grimoire/torch2trt_dynamic) v0.5.0 (*optional* for evaluating models without TensorRT, currently TensorRT optimization only supports devices with [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/), and already included in [JetPack SDK](https://developer.nvidia.com/embedded/Jetpack) if using Jetson devices):
   1. Install CUDA 10.2/11.4 and cuDNN 8.2.
   2. Download TensorRT 8.2.1.8 tar file [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install TensorRT (refer to [official documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/install-guide/index.html#installing-tar) for more details).
   ```Shell
   version="8.x.x.x"
   arch=$(uname -m)
   cuda="cuda-x.x"
   cudnn="cudnn8.x"
   tar xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.${cudnn}.tar.gz
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

   cd TensorRT-${version}/python
   python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
   
   cd TensorRT-${version}/uff
   python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

   cd TensorRT-${version}/graphsurgeon
   python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
   ```
   3. Install [torch2trt_dynamic](https://github.com/grimoire/torch2trt_dynamic).
   ```Shell
   git clone https://github.com/grimoire/torch2trt_dynamic.git torch2trt_dynamic
   cd torch2trt_dynamic 
   python setup.py develop
   ```
   4. Install deformable convolution module to pytorch if you want to work with yolact_edge+ models. Go to ./external/mod_def_conv and run setup.py
   ```Shell
   cd external/mod_def_conv
   python setup.py install
   ```
   5. Install [amirstan_plugin](https://github.com/grimoire/amirstan_plugin) which contain the deformable convolution plugin with dynamic shapes for TensorRT 8.x. IT is needed only if you want to work with yolact edge+ models.
   ```Shell
    apt install -y software-properties-common
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    apt update && apt install -y cmake
    git clone --depth=1 --branch v0.5.0 https://github.com/grimoire/amirstan_plugin.git
    cd amirstan_plugin
    cmake -DTENSORRT_DIR=/usr/lib/x86_64-linux-gnu -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    make -j$(nproc)
   
   export AMIRSTAN_LIBRARY_PATH=<amirstan_plugin_root>/lib
   ```


 - Install some other packages:
   ```Shell
   # Cython needs to be installed before pycocotools
   pip install cython
   pip install opencv-python pillow matplotlib
   pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
   pip install GitPython termcolor tensorboard packaging
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
