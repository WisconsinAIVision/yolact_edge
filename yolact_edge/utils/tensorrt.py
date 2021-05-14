import logging
import numpy as np
import torch
import cv2
from pathlib import Path
import math
import os

def convert_to_tensorrt(net, cfg, args, transform):
    logger = logging.getLogger("yolact.eval")

    torch2trt_flags = ['torch2trt_backbone', 'torch2trt_backbone_int8', 'torch2trt_protonet', 'torch2trt_protonet_int8', 'torch2trt_fpn', 'torch2trt_fpn_int8', 'torch2trt_prediction_module', 'torch2trt_prediction_module_int8', 'torch2trt_spa', 'torch2trt_spa_int8', 'torch2trt_flow_net', 'torch2trt_flow_net_int8']

    if args.disable_tensorrt:
        for key in torch2trt_flags:
            setattr(cfg, key, False)
    
    if args.use_fp16_tensorrt:
        for key in torch2trt_flags:
            if 'int8' in key and getattr(cfg, key, False):
                setattr(cfg, key, False)
                setattr(cfg, key[:-5], True)

    use_tensorrt_conversion = any(getattr(cfg, key, False) for key in torch2trt_flags)
    if use_tensorrt_conversion:
        logger.info("Converting to TensorRT...")
    else:
        return

    net.model_path = args.trained_model

    if args.use_tensorrt_safe_mode:
        cfg.use_tensorrt_safe_mode = True
        logger.warning("Running TensorRT in safe mode. This is an attempt to solve various TensorRT engine errors.")

    calibration_dataset = None
    calibration_protonet_dataset = None
    calibration_ph_dataset = None
    calibration_fpn_dataset = None
    calibration_flow_net_dataset = None

    if (cfg.torch2trt_backbone_int8 or cfg.torch2trt_protonet_int8 or cfg.torch2trt_flow_net_int8):
        if (not cfg.torch2trt_backbone_int8 or net.has_trt_cached_module('backbone', True)) and \
            (not cfg.torch2trt_protonet_int8 or net.has_trt_cached_module('proto_net', True)) and \
                (not cfg.torch2trt_flow_net_int8 or net.has_trt_cached_module('flow_net', True)):
            logger.debug('Skipping generation of calibration dataset for backbone/flow_net because there is cache...')
        else:
            logger.debug('Generating calibration dataset for backbone of {} images...'.format(cfg.torch2trt_max_calibration_images))

            calib_images = cfg.dataset.calib_images
            if args.calib_images is not None:
                calib_images = args.calib_images

            def pull_calib_dataset(calib_folder, transform=transform, max_calibration_images=cfg.torch2trt_max_calibration_images):
                images = []
                paths = [str(x) for x in Path(calib_folder).glob('*')]
                paths = paths[:max_calibration_images]
                for path in paths:
                    img = cv2.imread(path)
                    height, width, _ = img.shape

                    img, _, _, _ = transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                        {'num_crowds': 0, 'labels': np.array([0])})

                    images.append(torch.from_numpy(img).permute(2, 0, 1))

                calibration_dataset = torch.stack(images)
                if args.cuda:
                    calibration_dataset = calibration_dataset.cuda()
                return calibration_dataset

            if ':' in calib_images:
                calib_dir, prev_folder, next_folder = calib_images.split(':')
                prev_dir = os.path.join(calib_dir, prev_folder)
                next_dir = os.path.join(calib_dir, next_folder)

                calibration_dataset = pull_calib_dataset(prev_dir)
                calibration_next_dataset = pull_calib_dataset(next_dir)
            else:
                calibration_dataset = pull_calib_dataset(calib_images)

    n_images_per_batch = 1
    if cfg.torch2trt_protonet_int8:
        if net.has_trt_cached_module('proto_net', True):
            logger.debug('Skipping generation of calibration dataset for protonet because there is cache...')
        else:
            logger.debug('Generating calibration dataset for protonet with {} images...'.format(cfg.torch2trt_max_calibration_images))
            calibration_protonet_dataset = []

            def forward_hook(self, inputs, outputs):
                calibration_protonet_dataset.append(inputs[0])

            proto_net_handle = net.proto_net.register_forward_hook(forward_hook)

    if (cfg.torch2trt_protonet_int8 or cfg.torch2trt_flow_net_int8):
        if (not cfg.torch2trt_protonet_int8 or net.has_trt_cached_module('proto_net', True)) and (not cfg.torch2trt_flow_net_int8 or net.has_trt_cached_module('flow_net', True)):
            logger.debug('Skipping generation of calibration dataset for protonet/flow_net because there is cache...')
        else:
            with torch.no_grad():
                laterals = []
                f1, f2, f3 = [], [], []
                for i in range(math.ceil(cfg.torch2trt_max_calibration_images / n_images_per_batch)):
                    gt_forward_out = net(calibration_dataset[i*n_images_per_batch:(i+1)*n_images_per_batch], extras={
                        "backbone": "full",
                        "keep_statistics": True,
                        "moving_statistics": None
                    })
                    laterals.append(gt_forward_out["lateral"])
                    f1.append(gt_forward_out["feats"][0])
                    f2.append(gt_forward_out["feats"][1])
                    f3.append(gt_forward_out["feats"][2])

            laterals = torch.cat(laterals, dim=0)
            f1 = torch.cat(f1, dim=0)
            f2 = torch.cat(f2, dim=0)
            f3 = torch.cat(f3, dim=0)

    if cfg.torch2trt_protonet_int8:
        if net.has_trt_cached_module('proto_net', True):
            logger.debug('Skipping generation of calibration dataset for protonet because there is cache...')
        else:
            proto_net_handle.remove()
            calibration_protonet_dataset = torch.cat(calibration_protonet_dataset, dim=0)

    if cfg.torch2trt_flow_net_int8:
        if net.has_trt_cached_module('flow_net', True):
            logger.debug('Skipping generation of calibration dataset for flow_net because there is cache...')
        else:
            logger.debug('Generating calibration dataset for flow_net with {} images...'.format(cfg.torch2trt_max_calibration_images))
            calibration_flow_net_dataset = []

            def forward_hook(self, inputs, outputs):
                calibration_flow_net_dataset.append(inputs[0])

            handle = net.flow_net.flow_net.register_forward_hook(forward_hook)
            for i in range(math.ceil(cfg.torch2trt_max_calibration_images / n_images_per_batch)):
                extras = {
                    "backbone": "partial",
                    "moving_statistics": {
                        "lateral": laterals[i*n_images_per_batch:(i+1)*n_images_per_batch],
                        "feats": [
                            f1[i*n_images_per_batch:(i+1)*n_images_per_batch],
                            f2[i*n_images_per_batch:(i+1)*n_images_per_batch],
                            f3[i*n_images_per_batch:(i+1)*n_images_per_batch]
                        ]
                    }
                }
                with torch.no_grad():
                    net(calibration_next_dataset[i*n_images_per_batch:(i+1)*n_images_per_batch], extras=extras)
            handle.remove()

            calibration_flow_net_dataset = torch.cat(calibration_flow_net_dataset, dim=0)

    if cfg.torch2trt_backbone or cfg.torch2trt_backbone_int8:
        logger.info("Converting backbone to TensorRT...")
        net.to_tensorrt_backbone(cfg.torch2trt_backbone_int8, calibration_dataset=calibration_dataset, batch_size=args.trt_batch_size)

    if cfg.torch2trt_protonet or cfg.torch2trt_protonet_int8:
        logger.info("Converting protonet to TensorRT...")
        net.to_tensorrt_protonet(cfg.torch2trt_protonet_int8, calibration_dataset=calibration_protonet_dataset, batch_size=args.trt_batch_size)

    if cfg.torch2trt_fpn or cfg.torch2trt_fpn_int8:
        logger.info("Converting FPN to TensorRT...")
        net.to_tensorrt_fpn(cfg.torch2trt_fpn_int8, batch_size=args.trt_batch_size)

    if cfg.torch2trt_prediction_module or cfg.torch2trt_prediction_module_int8:
        logger.info("Converting PredictionModule to TensorRT...")
        net.to_tensorrt_prediction_head(cfg.torch2trt_prediction_module_int8, batch_size=args.trt_batch_size)

    if cfg.torch2trt_spa or cfg.torch2trt_spa_int8:
        logger.info('Converting SPA to TensorRT...')
        assert not cfg.torch2trt_spa_int8
        net.to_tensorrt_spa(cfg.torch2trt_spa_int8, batch_size=args.trt_batch_size)

    if cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8:
        logger.info('Converting flow_net to TensorRT...')
        net.to_tensorrt_flow_net(cfg.torch2trt_flow_net_int8, calibration_dataset=calibration_flow_net_dataset, batch_size=args.trt_batch_size)

    logger.info("Converted to TensorRT.")