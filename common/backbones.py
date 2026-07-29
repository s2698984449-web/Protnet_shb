"""Independent encoders for RGB (YOLO26) / thermal / ultrasonic modalities."""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM channel branch: avg/max pool + shared MLP."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial branch: channel-wise avg/max -> 7x7 conv."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


def _resolve_yolo_weights(weights_path, default_name='yolo26n.pt'):
    if weights_path and os.path.isfile(weights_path):
        return os.path.abspath(weights_path)

    search_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), 'weights'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '航空缺陷检测', 'weights')),
    ]
    for directory in search_dirs:
        candidate = os.path.join(directory, default_name)
        if os.path.isfile(candidate):
            return candidate

    if weights_path:
        return weights_path
    return default_name


class YOLO26BackboneEncoder(nn.Module):
    """YOLO26 backbone+neck (Detect head removed) -> 128-d embedding."""

    def __init__(self, out_dim=128, imgsz=640, weights_path='', use_cbam=False, cbam_reduction=16):
        super().__init__()
        from ultralytics import YOLO

        self.use_cbam = bool(use_cbam)
        path = _resolve_yolo_weights(weights_path)
        if os.path.isfile(path):
            logging.info('Loading YOLO26 backbone from: {}'.format(path))
            yolo = YOLO(path)
        else:
            logging.warning(
                'YOLO26 weights not found at {}. '
                'Building yolo26n from yaml (random init). '
                'Place yolo26n.pt under weights/ for pretrained backbone.'.format(path))
            yolo = YOLO('yolo26n.yaml')
        self.det_model = yolo.model
        self.det_idx = len(self.det_model.model) - 1
        self.imgsz = imgsz

        with torch.no_grad():
            feat = self._extract_features(torch.zeros(1, 3, imgsz, imgsz))
        if feat.dim() != 4:
            raise RuntimeError(
                'Unexpected YOLO26 feature shape {}. Check ultralytics version.'.format(
                    tuple(feat.shape)))
        in_ch = feat.shape[1]
        self.cbam = CBAM(in_ch, reduction=cbam_reduction) if self.use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, out_dim)
        self.out_dim = out_dim
        self.backbone_frozen = False
        extra = ', CBAM on' if self.use_cbam else ''
        logging.info('YOLO26 backbone ready: in_ch={}, out_dim={}, imgsz={}{}.'.format(
            in_ch, out_dim, imgsz, extra))

    def freeze_backbone(self):
        """Freeze YOLO26 conv backbone; keep proj + fusion meta-trainable."""
        self.backbone_frozen = True
        for p in self.det_model.parameters():
            p.requires_grad = False
        self.det_model.eval()
        logging.info('YOLO26 backbone frozen (proj layer still trainable in outer loop).')

    def _extract_features(self, x):
        y = []
        for m in self.det_model.model[:self.det_idx]:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.det_model.save else None)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        return x

    def encode(self, x):
        if x.shape[-2] != self.imgsz or x.shape[-1] != self.imgsz:
            x = F.interpolate(
                x, size=(self.imgsz, self.imgsz),
                mode='bilinear', align_corners=False)
        feat = self._extract_features(x)
        if self.backbone_frozen:
            feat = feat.detach()
        if self.cbam is not None:
            feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)

    def forward(self, x):
        return self.encode(x)


class ThermalEncoder(nn.Module):
    """Lightweight CNN for single-channel thermal images."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def encode(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)

    def forward(self, x):
        return self.encode(x)


class UltrasonicEncoder(nn.Module):
    """1D CNN for 512-point ultrasonic signals."""

    def __init__(self, out_dim=128, input_len=512):
        super().__init__()
        self.input_len = input_len
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 8, stride=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 4, stride=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 4, stride=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)

    def forward(self, x):
        return self.encode(x)
