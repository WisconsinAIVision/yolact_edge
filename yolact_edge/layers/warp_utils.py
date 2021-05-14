import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from yolact_edge.utils import timer

from yolact_edge.data.config import cfg

try:
    from torch2trt import torch2trt
    from torch2trt.torch2trt import TRTModule
    use_torch2trt = True
except:
    use_torch2trt = False

use_jit = False if use_torch2trt else torch.cuda.device_count() <= 1

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn
script_wrapper = torch.jit.script if use_jit else lambda fn, _rcn=None: fn


cache = {}

@torch.jit.ignore
def generate_grid_as(n: int, h: int, w: int, t: torch.Tensor):
    if (n, h, w) in cache:
        return cache[(n, h, w)].clone()

    x_ = torch.arange(w, dtype=t.dtype, device=t.device).view(1, -1).expand(h, -1)
    y_ = torch.arange(h, dtype=t.dtype, device=t.device).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)

    cache[(n, h, w)] = grid.clone()

    return grid.clone()


def tensor_as(x: float, t: torch.Tensor):
    return torch.tensor(x, dtype=t.dtype, device=t.device)


@script_wrapper
def deform_op(x, offset, mode: str = "bilinear", padding_mode: str = "zeros"):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
        padding_mode (str): 'zeros' or 'border'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == offset.size()[-2:]
    n, _, h, w = offset.size()
    grid = generate_grid_as(n, h, w, x)

    grid *= torch.tensor([2/(w-1), 2/(h-1)], dtype=torch.float, device=x.device).view(1, 2, 1, 1)
    grid -= 1
    # offset *= torch.tensor([384. / 512 / 16 * 2, 512. / 384 / 16 * 2], dtype=torch.float, device=x.device).view(1, 2, 1, 1)
    grid += offset / 8
    grid = grid.permute(0, 2, 3, 1)

    output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    return output
