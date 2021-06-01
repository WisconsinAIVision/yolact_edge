import numpy as np
import urllib
import time
import cv2
from yolact_edge.inference import YOLACTEdgeInference

weights = "yolact_edge_resnet50_54_800000.pth"
# All available model configs, depends on which weights
# you use. More info could be found in data/config.py.
model_configs = [
    'yolact_edge_mobilenetv2_config',
    'yolact_edge_vid_config',
    'yolact_edge_vid_minimal_config',
    'yolact_edge_vid_trainflow_config',
    'yolact_edge_youtubevis_config',
    'yolact_resnet50_config',
    'yolact_resnet152_config',
    'yolact_edge_resnet50_config',
    'yolact_edge_vid_resnet50_config',
    'yolact_edge_vid_trainflow_resnet50_config',
    'yolact_edge_youtubevis_resnet50_config',
]
config = model_configs[5]
# All available model datasets, depends on which weights
# you use. More info could be found in data/config.py.
datasets = [
    'coco2014_dataset',
    'coco2017_dataset',
    'coco2017_testdev_dataset',
    'flying_chairs_dataset',
    'youtube_vis_dataset',
]
dataset = datasets[1]
# Used tensorrt calibration
calib_images = "./data/calib_images"
# Override some default configuration
config_ovr = {
    'use_fast_nms': True,  # Does not work with regular nms
    'mask_proto_debug': False
}
model_inference = YOLACTEdgeInference(
    weights, config, dataset, calib_images, config_ovr)

img = None

try:
    with urllib.request.urlopen("http://images.cocodataset.org/val2017/000000439715.jpg") as f:
        img = np.asarray(bytearray(f.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
except:
    pass

if img is None:
    print("Couldn't retrieve image for benchmark...")
    exit(1)

print("Benchmarking performance...")
start = time.time()
samples = 200
for i in range(samples):
    p = model_inference.predict(img, False)
print(f"Average {1 / ( (time.time() - start) / samples )} FPS")
