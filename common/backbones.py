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


class CalibratedFeatureFusion(nn.Module):
    """CFF: multi-scale feature calibration + fusion.

    Pipeline:
      1. Channel align  — 每个尺度 → embed_dim 通道（1×1 conv）
      2. Spatial align  — 所有尺度上/下采样到目标分辨率
      3. Attention weighting — 每尺度 GAP → softmax → 加权求和
      4. CBAM          — 融合后特征再做 CBAM（在 encoder 层面做）
    """

    def __init__(self, scale_channels, embed_dim=128, target_scale_idx=1):
        super().__init__()
        self.num_scales = len(scale_channels)
        self.target_scale_idx = target_scale_idx
        self.scale_names = list(scale_channels.keys())
        self.embed_dim = embed_dim

        self.channel_align = nn.ModuleDict()
        for name, ch in scale_channels.items():
            self.channel_align[name] = nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(),
            )

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(embed_dim * self.num_scales, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.num_scales),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, multi_scale_feats):
        """multi_scale_feats: dict {name: (B, C, H, W)}."""
        aligned = {}
        target_size = None

        for name in self.scale_names:
            feat = self.channel_align[name](multi_scale_feats[name])
            aligned[name] = feat
            if name == self.scale_names[self.target_scale_idx]:
                target_size = (feat.shape[2], feat.shape[3])

        resized = []
        for name in self.scale_names:
            f = aligned[name]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size,
                                  mode='bilinear', align_corners=False)
            resized.append(f)

        concat = torch.cat(resized, dim=1)
        weights = F.softmax(self.scale_attention(concat), dim=-1)
        w = weights.view(weights.shape[0], weights.shape[1], 1, 1)
        weighted_sum = sum(w[:, i:i + 1, :, :] * r for i, r in enumerate(resized))
        return self.fuse_conv(weighted_sum), None


class YOLO26BackboneEncoder(nn.Module):
    """YOLO26 backbone+neck -> 128-d embedding.

    Dual-mode (controlled by use_multiscale_cff):
      False (default): 原始单尺度模式 — 只取最后一层特征
      True:            多尺度 CFF 模式 — P2/P3/P4 三层特征融合 + CBAM
    """

    # YOLO26 neck layer indices → (name, channels, downsample_ratio)
    NECK_SAVE_IDX = {16: ('p2', 128, 8), 19: ('p3', 128, 16), 22: ('p4', 256, 32)}
    # 多尺度融合用 p2/p3/p4，即 [16, 19, 22]
    MULTI_SCALE_IDX = [16, 19, 22]
    # 融合目标分辨率: p3 (index 19) 的分辨率
    FUSION_TARGET_IDX = 1  # refers to MULTI_SCALE_IDX[1] = p3

    def __init__(self, out_dim=128, imgsz=640, weights_path='',
                 use_cbam=False, cbam_reduction=16,
                 use_multiscale_cff=False, fusion_embed_dim=None):
        super().__init__()
        from ultralytics import YOLO

        self.use_multiscale_cff = bool(use_multiscale_cff)
        self.use_cbam_after_fusion = bool(use_cbam)
        self.fusion_embed_dim = fusion_embed_dim or out_dim
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

        self.out_dim = out_dim

        # 注册 hooks 在 dummy forward 之前，这样 dummy forward 也能捕获特征
        self._hook_handles = []
        self._hook_outputs = {}
        self._register_hooks()

        with torch.no_grad():
            feat = self._extract_single_scale(torch.zeros(1, 3, imgsz, imgsz))
        if feat.dim() != 4:
            raise RuntimeError(
                'Unexpected YOLO26 feature shape {}. Check ultralytics version.'.format(
                    tuple(feat.shape)))
        in_ch = feat.shape[1]

        if self.use_multiscale_cff:
            self._setup_multiscale_path(in_ch)
        elif self.use_cbam_after_fusion:
            self.cbam = CBAM(in_ch, reduction=cbam_reduction)
            self.proj = nn.Linear(in_ch, out_dim)
        else:
            self.cbam = None
            self.proj = nn.Linear(in_ch, out_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_frozen = False

        mode_str = 'CFF multi-scale' if self.use_multiscale_cff else 'single-scale'
        cbam_str = '+CBAM' if self.use_cbam_after_fusion else ''
        logging.info(
            'YOLO26 backbone ready: in_ch={}, out_dim={}, imgsz={}, mode={}{}.'.format(
                in_ch, out_dim, imgsz, mode_str, cbam_str))

    def _setup_multiscale_path(self, in_ch):
        scale_channels = {
            'p2': 64,   # layer 16: 64ch, 80x80 (after neck concat)
            'p3': 128,  # layer 19: 128ch, 40x40
            'p4': 256,  # layer 22: 256ch, 20x20
        }
        self.cff = CalibratedFeatureFusion(
            scale_channels=scale_channels,
            embed_dim=self.fusion_embed_dim,
            target_scale_idx=self.FUSION_TARGET_IDX,
        )
        if self.use_cbam_after_fusion:
            self.cbam = CBAM(self.fusion_embed_dim, reduction=max(self.fusion_embed_dim // 16, 1))
        else:
            self.cbam = None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.fusion_embed_dim, self.out_dim)

    def freeze_backbone(self):
        """Freeze YOLO26 conv backbone; keep proj + fusion meta-trainable."""
        self.backbone_frozen = True
        for p in self.det_model.parameters():
            p.requires_grad = False
        self.det_model.eval()
        logging.info('YOLO26 backbone frozen.')

    def _register_hooks(self):
        """Register forward hooks to capture multi-scale outputs cleanly."""
        self._hook_outputs = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, (list, tuple)):
                    out = output[0]
                else:
                    out = output
                self._hook_outputs[name] = out
            return hook_fn

        for idx, (name, _, _) in self.NECK_SAVE_IDX.items():
            layer = list(self.det_model.model[:self.det_idx])[idx]
            h = layer.register_forward_hook(make_hook(name))
            self._hook_handles.append(h)

    def _extract_single_scale(self, x):
        """单尺度：清空 hook 缓存，跑 forward，取最后一层 save 特征。"""
        self._hook_outputs.clear()
        self.det_model(x)
        last_name = list(self.NECK_SAVE_IDX.values())[-1][0]
        return self._hook_outputs.get(last_name)

    def _extract_multiscale(self, x):
        """多尺度：清空 hook 缓存，跑 forward，取所有 save 层特征。"""
        self._hook_outputs.clear()
        self.det_model(x)
        return {name: self._hook_outputs[name]
                for name, _, _ in self.NECK_SAVE_IDX.values()
                if name in self._hook_outputs}

    def encode(self, x):
        was_3d = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            was_3d = True
        elif x.dim() < 3:
            raise ValueError(f"Expected at least 3D tensor, got {x.dim()}D")
        if x.shape[-2] != self.imgsz or x.shape[-1] != self.imgsz:
            x = F.interpolate(x, size=(self.imgsz, self.imgsz),
                              mode='bilinear', align_corners=False)

        if self.use_multiscale_cff:
            scales = self._extract_multiscale(x)
            if self.backbone_frozen:
                scales = {k: v.detach() for k, v in scales.items()}
            fused, _ = self.cff(scales)
        else:
            feat = self._extract_single_scale(x)
            if self.backbone_frozen:
                feat = feat.detach()
            fused = feat

        if self.cbam is not None:
            fused = self.cbam(fused)
        feat = self.pool(fused).flatten(1)
        out = self.proj(feat)
        if was_3d:
            out = out.squeeze(0)
        return out

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


class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone for RGB images.

    Loads pretrained ImageNet weights automatically from torchvision.
    Supports both frozen (feature extractor) and unfrozen (meta-trainable) modes.
    """

    def __init__(self, out_dim=128, pretrained=True, freeze_backbone=False, weights_path=None):
        super().__init__()
        import torchvision.models as models

        if weights_path:
            state_dict = torch.load(weights_path, map_location='cpu')
            resnet = models.resnet18(weights=None)
            resnet.load_state_dict(state_dict)
        else:
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = models.resnet18(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.out_dim = out_dim
        self.feat_dim = 512

        self.proj = nn.Linear(self.feat_dim, out_dim)
        self.backbone_frozen = False

        if freeze_backbone:
            self.freeze_backbone()

        mode = 'frozen' if self.backbone_frozen else 'unfrozen'
        logging.info(f'ResNet-18 ready: pretrained={pretrained}, mode={mode}, out_dim={out_dim}.')

    def freeze_backbone(self):
        """Freeze all conv layers; only proj (out_dim projection) is trainable."""
        self.backbone_frozen = True
        for module in [self.conv1, self.bn1, self.layer1, self.layer2,
                        self.layer3, self.layer4]:
            for p in module.parameters():
                p.requires_grad = False
        logging.info('ResNet-18 backbone frozen (only proj trainable).')

    def encode(self, x):
        was_3d = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            was_3d = True
        elif x.dim() < 3:
            raise ValueError(f'Expected at least 3D tensor, got {x.dim()}D')

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3, :, :]

        if self.backbone_frozen:
            with torch.no_grad():
                out = self._forward_impl(x)
        else:
            out = self._forward_impl(x)

        out = self.proj(out)
        if was_3d:
            out = out.squeeze(0)
        return out

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self.encode(x)
