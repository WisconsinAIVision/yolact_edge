import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from .config import cfg, MEANS, STD
from pycocotools import mask as maskUtils
import contextlib
import io
import logging
import time


def collate_fn_flying_chairs(batch):
    imgs_1 = []
    imgs_2 = []
    flows = []

    for sample in batch:
        imgs_1.append(sample[0])
        imgs_2.append(sample[1])
        flows.append(sample[2])

    return torch.stack(imgs_1, 0), torch.stack(imgs_2, 0), torch.stack(flows, 0)


class FlyingChairs(data.Dataset):
    """`YoutubeVIS <https://youtube-vos.org/dataset/vis/>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, is_train=True):
        # Do this here because we have too many things named COCO

        self.root = image_path

        with open(info_file, "r") as file:
            res = file.read()
        ids = res.split('\n')
        ids = [int(x) for x in ids if x]
        keep_label = 1 if is_train else 2
        ids = {idx: x for idx, x in enumerate(ids) if x == keep_label}
        self.ids = list(ids.keys())

    def __getitem__(self, index):
        flow_id = self.ids[index] + 1
        img1_path = os.path.join(self.root, "{:05d}_img1.ppm".format(flow_id))
        img2_path = os.path.join(self.root, "{:05d}_img2.ppm".format(flow_id))
        flow_path = os.path.join(self.root, "{:05d}_flow.flo".format(flow_id))

        img1 = self.readImage(img1_path)
        img2 = self.readImage(img2_path)
        flow = self.readFlow(flow_path)

        h, w, _ = img1.shape

        flow = flow * 2 / np.array([w, h]) * 8

        target_size = (550, 550) # FIXME: hard code image size

        img1 = cv2.resize(img1, target_size)
        img2 = cv2.resize(img2, target_size)
        flow = cv2.resize(flow, target_size)

        img1 = (img1 - MEANS) / STD
        img2 = (img2 - MEANS) / STD

        img1 = img1[:, :, ::-1]
        img2 = img2[:, :, ::-1]

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        flow = flow.astype(np.float32)

        t = transforms.ToTensor()
        return t(img1), t(img2), t(flow)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def readFlow(name):
        f = open(name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        return flow.astype(np.float32)

    @staticmethod
    def readImage(name):
        return cv2.imread(name)
