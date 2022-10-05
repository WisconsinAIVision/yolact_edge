import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
import mod_dcn_op_v2



class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, mask, weight, bias, stride, padding,
                 dilation, groups, deform_groups):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            'ai.onnx.contrib::ModulatedDeformableConvPluginDynamic',
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            # groups_i=groups,
            group_i=groups,
            # deform_groups_i=deform_groups
            deformable_group_i=deform_groups)

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deform_groups=1):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        mod_dcn_op_v2.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        mod_dcn_op_v2.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): bias if exists
    """

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConv2dPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


if __name__ == "__main__":
    from torch.autograd import gradcheck

    groups = 1
    deformable_groups = 1
    N, inC, inH, inW = 2, 2, 4, 4
    outC = 2
    kH, kW = 3, 3

    def check_gradient_dconv():
        t = torch.randn(N, inC, inH, inW, dtype=torch.double).cuda()
        # t = torch.randn(N, inC, inH, inW).cuda()
        t.requires_grad = True

        offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW, dtype=torch.double).cuda()
        # offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH, inW).cuda()
        # offset.data.zero_()
        # offset.data -= 0.5
        offset.requires_grad = True

        mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW, dtype=torch.double).cuda()
        # mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
        # mask.data.zero_()
        mask.requires_grad = True
        mask = torch.sigmoid(mask)

        weight = torch.randn(outC, inC, kH, kW, dtype = torch.double).cuda()
        # weight = torch.randn(outC, inC, kH, kW).cuda()
        weight.requires_grad = True

        bias = torch.rand(outC, dtype=torch.double).cuda()
        # bias = torch.rand(outC).cuda()
        bias.requires_grad = True

        func = ModulatedDeformConv2dFunction.apply

        gradcheck(func, (t, offset, mask, weight, bias,  (1,1), (1,1), (1,1), groups, deformable_groups), eps=1e-3, atol=1e-3, rtol=1e-2)
    check_gradient_dconv()


