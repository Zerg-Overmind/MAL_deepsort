# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, top_blocks=None,
                 use_gn=False):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        # If in_channels is 0, it would be used. 
        self.valid_layers = [i > 0 for i in in_channels_list]
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue

            if use_gn:
                inner_block_module = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels))
                layer_block_module = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.GroupNorm(32, out_channels))
            else:
                inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
                layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

            for module in [inner_block_module, layer_block_module]:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        # Caffe2 implementation uses XavierFill, which in fact
                        # corresponds to kaiming_uniform_ in PyTorch
                        nn.init.kaiming_uniform_(m.weight, a=1)
                        nn.init.constant_(m.bias, 0)
                    if isinstance(m, nn.GroupNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0)

            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if len(inner_block):
                inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
                inner_lateral = getattr(self, inner_block)(feature)
                # TODO use size instead of scale to make it robust to different sizes
                # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
                # mode='bilinear', align_corners=False)
                last_inner = inner_lateral + inner_top_down
                tmp = getattr(self, layer_block)(last_inner)
                results.insert(0, tmp)
                # np.save(os.path.join('/workspace/retinanet/debug', '{}.npy'.format(inner_block)), tmp.cpu().numpy())

        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        p6 = self.p6(x)
        # np.save(os.path.join('/workspace/retinanet/debug', 'p6.npy'), p6.cpu().numpy())
        # np.save(os.path.join('/workspace/retinanet/debug', 'p6_weight.npy'), self.p6.weight.cpu().numpy())
        p7 = self.p7(F.relu(p6))
        # np.save(os.path.join('/workspace/retinanet/debug', 'p7.npy'), p7.cpu().numpy())
        return [p6, p7]