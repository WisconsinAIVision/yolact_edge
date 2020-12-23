import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OpticalFlowLoss(nn.Module):
    def __init__(self):
        super(OpticalFlowLoss, self).__init__()


    def forward(self, preds, gt):
        losses = {}
        loss_F = 0
        for pred in preds:
            _, _, h, w = pred.size()
            gt_downsample = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)
            loss_F += torch.norm(pred - gt_downsample, dim=1).mean()
        losses['F'] = loss_F
        return losses