import os.path
import io
import math
import torch
import torch.nn as nn
import numpy as np

from .backbones.backbone import build_backbone
from ._C import Engine
from .box import generate_anchors, snap_to_anchors, decode, nms
from .loss import FocalLoss, SmoothL1Loss



strides = [8, 16, 32, 64, 128]
# base anchor in maskrcnn-benchmark
cell_anchors = [[[-18.0000,  -8.0000,  25.0000,  15.0000], [-23.7183, -11.1191,  30.7183,  18.1191], [-30.9228, -15.0488,  37.9228,  22.0488], \
                 [-12.0000, -12.0000,  19.0000,  19.0000], [-16.1587, -16.1587,  23.1587,  23.1587], [-21.3984, -21.3984,  28.3984,  28.3984], \
                 [ -8.0000, -20.0000,  15.0000,  27.0000], [-11.1191, -26.2381,  18.1191,  33.2381], [-15.0488, -34.0976,  22.0488,  41.0976]], \
                [[-38.0000, -16.0000, 53.0000, 31.0000], [-49.9564, -22.2381, 64.9564, 37.2381], [-65.0204, -30.0976, 80.0204, 45.0976], \
                 [-24.0000, -24.0000, 39.0000, 39.0000], [-32.3175, -32.3175, 47.3175, 47.3175], [-42.7968, -42.7968, 57.7968, 57.7968], \
                 [-14.0000, -36.0000, 29.0000, 51.0000], [-19.7183, -47.4365, 34.7183, 62.4365], [-26.9228, -61.8456, 41.9228, 76.8456]], \
                [[-74.0000, -28.0000, 105.0000, 59.0000], [-97.3929, -39.4365, 128.3929, 70.4365], [-126.8661, -53.8456, 157.8661, 84.8456], \
                 [-48.0000, -48.0000, 79.0000, 79.0000], [-64.6349, -64.6349, 95.6349, 95.6349], [-85.5937, -85.5937, 116.5937, 116.5937], \
                 [-30.0000, -76.0000, 61.0000, 107.0000], [-41.9564, -99.9127, 72.9564, 130.9127], [-57.0204, -130.0409, 88.0204, 161.0409]], \
                [[-150.0000, -60.0000, 213.0000, 123.0000], [-197.3056, -83.9127, 260.3056, 146.9127], [-256.9070, -114.0409, 319.9070, 177.0409], \
                 [-96.0000, -96.0000, 159.0000, 159.0000], [-129.2699, -129.2699, 192.2699, 192.2699], [-171.1873, -171.1873, 234.1873, 234.1873], \
                 [-58.0000, -148.0000, 121.0000, 211.0000], [-81.3929, -194.7858, 144.3929, 257.7858], [-110.8661, -253.7322, 173.8661, 316.7322]], \
                [[-298.0000, -116.0000, 425.0000, 243.0000], [-392.0914, -162.7858, 519.0914, 289.7858], [-510.6392, -221.7322, 637.6392, 348.7322], \
                 [-192.0000, -192.0000, 319.0000, 319.0000], [-258.5398, -258.5398, 385.5398, 385.5398], [-342.3747, -342.3747, 469.3747, 469.3747], \
                 [-118.0000, -300.0000, 245.0000, 427.0000], [-165.3056, -394.6113, 292.3056, 521.6113], [-224.9070, -513.8140, 351.9070, 640.8140]] \
                ]
"""
# base anchor in retinanet-example
cell_anchors = [[[-12.0000, -12.0000,  19.0000,  19.0000],
                 [ -8.0000, -20.0000,  15.0000,  27.0000],
                 [-18.0000,  -8.0000,  25.0000,  15.0000],
                 [-16.1587, -16.1587,  23.1587,  23.1587],
                 [-11.1191, -26.2381,  18.1191,  33.2381],
                 [-23.7183, -11.1191,  30.7183,  18.1191],
                 [-21.3984, -21.3984,  28.3984,  28.3984],
                 [-15.0488, -34.0976,  22.0488,  41.0976],
                 [-30.9228, -15.0488,  37.9228,  22.0488]], \
                [[-24.0000, -24.0000, 39.0000, 39.0000],
                 [-14.0000, -36.0000, 29.0000, 51.0000],
                 [-38.0000, -16.0000, 53.0000, 31.0000],
                 [-32.3175, -32.3175, 47.3175, 47.3175],
                 [-19.7183, -47.4365, 34.7183, 62.4365],
                 [-49.9564, -22.2381, 64.9564, 37.2381],
                 [-42.7968, -42.7968, 57.7968, 57.7968],
                 [-26.9228, -61.8456, 41.9228, 76.8456],
                 [-65.0204, -30.0976, 80.0204, 45.0976]], \
                [[-48.0000, -48.0000, 79.0000, 79.0000],
                 [-30.0000, -76.0000, 61.0000, 107.0000],
                 [-74.0000, -28.0000, 105.0000, 59.0000],
                 [-64.6349, -64.6349, 95.6349, 95.6349],
                 [-41.9564, -99.9127, 72.9564, 130.9127],
                 [-97.3929, -39.4365, 128.3929, 70.4365],
                 [-85.5937, -85.5937, 116.5937, 116.5937],
                 [-57.0204, -130.0409, 88.0204, 161.0409],
                 [-126.8661, -53.8456, 157.8661, 84.8456]], \
                [[-96.0000, -96.0000, 159.0000, 159.0000],
                 [-58.0000, -148.0000, 121.0000, 211.0000],
                 [-150.0000, -60.0000, 213.0000, 123.0000],
                 [-129.2699, -129.2699, 192.2699, 192.2699],
                 [-81.3929, -194.7858, 144.3929, 257.7858],
                 [-197.3056, -83.9127, 260.3056, 146.9127],
                 [-171.1873, -171.1873, 234.1873, 234.1873],
                 [-110.8661, -253.7322, 173.8661, 316.7322],
                 [-256.9070, -114.0409, 319.9070, 177.0409]], \
                [[-192.0000, -192.0000, 319.0000, 319.0000],
                 [-118.0000, -300.0000, 245.0000, 427.0000],
                 [-298.0000, -116.0000, 425.0000, 243.0000],
                 [-258.5398, -258.5398, 385.5398, 385.5398],
                 [-165.3056, -394.6113, 292.3056, 521.6113],
                 [-392.0914, -162.7858, 519.0914, 289.7858],
                 [-342.3747, -342.3747, 469.3747, 469.3747],
                 [-224.9070, -513.8140, 351.9070, 640.8140],
                 [-510.6392, -221.7322, 637.6392, 348.7322]] \
                ]
"""

class Model(nn.Module):
    'RetinaNet - https://arxiv.org/abs/1708.02002'

    def __init__(self, cfg, classes=80, config={}, ):
        super().__init__()

        self.cfg = cfg.clone()
        self.backbones = build_backbone(cfg)
        self.name = 'RetinaNet'
        self.exporting = False

        self.ratios = [1.0, 2.0, 0.5]
        self.scales = [4 * 2**(i/3) for i in range(3)]
        self.anchors = {}
        self.classes = classes

        self.threshold  = config.get('threshold', 0.05)
        self.top_n      = config.get('top_n', 1000)
        self.nms        = config.get('nms', 0.5)
        self.detections = config.get('detections', 100)

        self.stride = 32

        # classification and box regression heads
        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        anchors = len(self.ratios) * len(self.scales)
        self.cls_head = make_head(classes * anchors)
        self.box_head = make_head(4 * anchors)

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return '\n'.join([
            '     model: {}'.format(self.name),
            '  backbone: {}'.format(''.join(self.cfg.MODEL.BACKBONE.CONV_BODY)),
            '   classes: {}, anchors: {}'.format(self.classes, len(self.ratios) * len(self.scales)),
        ])

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbone(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        # Initialize class head prior
        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)
        self.cls_head[-1].apply(initialize_prior)

    def forward(self, x):
        if self.training: x, targets = x

        # Backbones forward pass
        features = self.backbones(x)
        """
        for idx, feat in enumerate(features):
            print('id: {}, fpn layer: {}, shape: {}'.format(ids, idx, feat.shape))
            np.save(os.path.join('/workspace/retinanet/debug', 'feature_{}_{}.npy'.format(ids,idx)), feat.cpu().numpy())
        """

        # Heads forward pass
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        """
        for idx, feat in enumerate(features):
            np.save(os.path.join('/workspace/retinanet/debug', 'cls_head_{}_{}.npy'.format(ids, idx)), cls_heads[idx].cpu().numpy())
            np.save(os.path.join('/workspace/retinanet/debug', 'box_head_{}_{}.npy'.format(ids, idx)), box_heads[idx].cpu().numpy())
        """

        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]
        # for idx, feat in enumerate(features):
        #     np.save(os.path.join('/workspace/retinanet/debug', 'cls_head_{}_{}.npy'.format(ids, idx)), cls_heads[idx].cpu().numpy())
      
        if self.exporting:
            self.strides = [x.shape[-1] // cls_head.shape[-1] for cls_head in cls_heads]
            return cls_heads, box_heads
    
        # Inference post-processing
        decoded = []
        for stride, cls_head, box_head in zip(strides, cls_heads, box_heads):
            # Generate level's anchors
            # stride = x.shape[-1] // cls_head.shape[-1]
            # print(stride)
            # if stride not in self.anchors:
            #     tmp = generate_anchors(stride, self.ratios, self.scales)
            #     self.anchors[stride] = tmp
            self.anchors[stride] = torch.Tensor(cell_anchors[strides.index(stride)])

            # Decode and filter boxes
            decoded.append(decode(cls_head, box_head, stride,
                self.threshold, self.top_n, self.anchors[stride]))

        # Perform non-maximum suppression
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)

    def _extract_targets(self, targets, stride, size):
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, -1] > -1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            snapped = snap_to_anchors(
                target, [s * stride for s in size[::-1]], stride,
                self.anchors[stride].to(targets.device), self.classes, targets.device)
            for l, s in zip((cls_target, box_target, depth), snapped): l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def save(self, state):
        checkpoint = {
            'backbone': [k for k, _ in self.backbones.items()],
            'classes': self.classes,
            'state_dict': self.state_dict()
        }

        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in state:
                checkpoint[key] = state[key]

        torch.save(checkpoint, state['path'])

    @classmethod
    def load(cls, cfg):
        print("cfg.MODEL.WEIGHT",cfg.MODEL.WEIGHT)
        #if not os.path.isfile(cfg.MODEL.WEIGHT):
        #    raise ValueError('No checkpoint {}'.format(cfg.MODEL.WEIGHT)
        model_state_dict = torch.load(cfg.MODEL.WEIGHT)
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(cfg, classes=cfg.RETINANET.NUM_CLASSES-1)
        # np.save(os.path.join('/workspace/retinanet/debug', 'stem_conv_weight_load_dict.npy'), model_state_dict['backbones.body.stem.conv1.weight'].data.cpu().numpy())
        model.load_state_dict(model_state_dict, strict=True)
        # np.save(os.path.join('/workspace/retinanet/debug', 'stem_conv_weight_load.npy'), model.backbones.body.stem.conv1.weight.cpu().numpy())

        return model

    def export(self, size, batch, precision, calibration_files, calibration_table, verbose, onnx_only=False):

        import torch.onnx.symbolic_opset9 as onnx_symbolic
        #import pdb
        #pdb.set_trace()
        def upsample_nearest2d(g, input, output_size):
            # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
            # needed to support dynamic upsampling ONNX forumlation
            # Here we hardcode scale=2 as a temporary workaround
            scales = g.op("Constant", value_t=torch.tensor([1.,1.,2.,2.]))
            return g.op("Upsample", input, scales, mode_s="nearest")

        onnx_symbolic.upsample_nearest2d = upsample_nearest2d

        # Export to ONNX
        print('Exporting to ONNX...')
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([1, 3, *size]).cuda()
        extra_args = { 'verbose': verbose }
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes, *extra_args)
        self.exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        model_name = '_'.join([k for k, _ in self.backbones.items()])
        anchors = [generate_anchors(stride, self.ratios, self.scales).view(-1).tolist()
            for stride in self.strides]
        return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision,
            self.threshold, self.top_n, anchors, self.nms, self.detections, calibration_files, model_name, calibration_table, verbose)

