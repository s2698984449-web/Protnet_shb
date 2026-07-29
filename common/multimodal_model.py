"""Full multimodal model: encoders + fusion + task encoder + stop head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .backbones import YOLO26BackboneEncoder, ThermalEncoder, UltrasonicEncoder
from .fusion import AsymmetricFusion
from .knowledge_base import DEFAULT_FUSION_PRIOR, fusion_prior_tensor
from .protonet_loop import MahalanobisHead
from .task_encoder import TaskEncoder, meta_vectors_to_tensors

COLLECT = 0
CLASSIFY = 1


class ValueBaseline(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state)


class AviationDefectModel(nn.Module):
    def __init__(self, num_classes=7, embed_dim=128, yolo_weights='', yolo_imgsz=640,
                 rgb_cbam=False):
        super().__init__()
        self.num_classes = num_classes
        self.rgb_cbam = bool(rgb_cbam)
        self.rgb_encoder = YOLO26BackboneEncoder(
            out_dim=embed_dim, imgsz=yolo_imgsz, weights_path=yolo_weights,
            use_cbam=self.rgb_cbam)
        self.thermal_encoder = ThermalEncoder(out_dim=embed_dim)
        self.us_encoder = UltrasonicEncoder(out_dim=embed_dim)
        self.task_encoder = TaskEncoder()
        self.fusion = AsymmetricFusion(embed_dim=embed_dim, num_classes=num_classes)
        self._embed_dim = embed_dim
        self.stop_net = nn.Sequential(
            nn.Linear(self.state_dim(), 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.mahal_head = MahalanobisHead(
            feat_dim=self.embed_dim() * 3, ways=num_classes)

    def embed_dim(self):
        return self._embed_dim

    def state_dim(self):
        d = self.embed_dim()
        return d * 3 + self.num_classes + 3 + 1

    def encode_modalities(self, rgb, thermal, ultrasonic):
        return (
            self.rgb_encoder.encode(rgb),
            self.thermal_encoder.encode(thermal),
            self.us_encoder.encode(ultrasonic),
        )

    def encode(self, rgb, thermal, ultrasonic, domain_meta=None, device=None):
        return self.encode_modalities(rgb, thermal, ultrasonic)

    def embed(self, rgb, thermal, ultrasonic, domain_meta=None):
        """Fixed-size embedding for ProtoNet (concatenated modality features)."""
        f_rgb, f_thermal, f_us = self.encode_modalities(rgb, thermal, ultrasonic)
        return torch.cat([f_rgb, f_thermal, f_us], dim=-1)

    def task_embedding(self, domain_meta, device):
        material, process, history = meta_vectors_to_tensors(domain_meta, device)
        return self.task_encoder(material, process, history)

    def kb_prior_batch(self, domain_meta, batch_size, device):
        material_id = domain_meta.get('material_id', 'AL7075')
        if 'fusion_prior' in domain_meta:
            prior = torch.tensor(domain_meta['fusion_prior'], dtype=torch.float32, device=device)
        else:
            prior = fusion_prior_tensor(material_id).to(device)
        return prior.unsqueeze(0).expand(batch_size, -1)

    def forward(self, rgb, thermal, ultrasonic, domain_meta=None, device=None):
        if device is None:
            device = rgb.device
        f_rgb, f_thermal, f_us = self.encode_modalities(rgb, thermal, ultrasonic)
        task_emb = None
        kb_prior = None
        if domain_meta is not None:
            task_emb = self.task_embedding(domain_meta, device)
            kb_prior = self.kb_prior_batch(domain_meta, rgb.size(0), device)
        fused, dynamic_w, logits_tuple = self.fusion(
            f_rgb, f_thermal, f_us, kb_prior=kb_prior, task_emb=task_emb)
        return fused

    def forward_logits(self, rgb, thermal, ultrasonic, domain_meta=None):
        device = rgb.device
        f_rgb, f_thermal, f_us = self.encode_modalities(rgb, thermal, ultrasonic)
        task_emb = None
        kb_prior = None
        if domain_meta is not None:
            task_emb = self.task_embedding(domain_meta, device)
            kb_prior = self.kb_prior_batch(domain_meta, rgb.size(0), device)
        return self.fusion(f_rgb, f_thermal, f_us, kb_prior=kb_prior, task_emb=task_emb)

    def build_state(self, feats, fused_logits, dynamic_w, support_depth_ratio=1.0):
        f_rgb, f_thermal, f_us = feats
        probs = F.softmax(fused_logits, dim=-1)
        return torch.cat([
            f_rgb, f_thermal, f_us,
            probs,
            dynamic_w,
            torch.full((f_rgb.size(0), 1), support_depth_ratio, device=f_rgb.device),
        ], dim=-1)

    def stop_logits(self, state):
        return self.stop_net(state)

    def classifier_parameters(self):
        enc = (
            list(self.rgb_encoder.parameters()) +
            list(self.thermal_encoder.parameters()) +
            list(self.us_encoder.parameters()) +
            list(self.task_encoder.parameters()) +
            list(self.fusion.parameters())
        )
        return enc

    def adapt_parameters(self, fusion_only=False):
        """Inner-loop parameters (Phase 1: fusion head only)."""
        if fusion_only:
            return list(self.fusion.parameters())
        return self.classifier_parameters()

    def meta_train_parameters(self, fusion_only=False):
        """Outer-loop meta-learnable parameters (flat list)."""
        groups = self.meta_optimizer_param_groups(
            fusion_only=fusion_only,
            backbone_meta_lr=1e-5,
            head_meta_lr=1e-3,
            meta_lr=1e-3,
        )
        flat = []
        for group in groups:
            flat.extend(group['params'])
        return flat

    def meta_optimizer_param_groups(
            self, fusion_only=False, backbone_meta_lr=1e-5,
            head_meta_lr=1e-3, meta_lr=1e-3):
        """Adam param groups for outer loop (Phase 1 ANIL vs unfreeze)."""
        if fusion_only:
            params = list(self.rgb_encoder.proj.parameters())
            if self.rgb_encoder.cbam is not None:
                params = list(self.rgb_encoder.cbam.parameters()) + params
            params += list(self.fusion.parameters())
            return [{'params': params, 'lr': meta_lr}]

        backbone = list(self.rgb_encoder.det_model.parameters())
        head = (
            list(self.rgb_encoder.proj.parameters()) +
            list(self.thermal_encoder.parameters()) +
            list(self.us_encoder.parameters()) +
            list(self.task_encoder.parameters()) +
            list(self.fusion.parameters()) +
            list(self.stop_net.parameters()) +
            list(self.mahal_head.parameters())
        )
        if self.rgb_encoder.cbam is not None:
            head = list(self.rgb_encoder.cbam.parameters()) + head
        return [
            {'params': backbone, 'lr': backbone_meta_lr},
            {'params': head, 'lr': head_meta_lr},
        ]

    def set_backbone_frozen(self, frozen=True):
        if frozen:
            self.rgb_encoder.freeze_backbone()
        else:
            self.rgb_encoder.backbone_frozen = False
            for p in self.rgb_encoder.det_model.parameters():
                p.requires_grad = True
            self.rgb_encoder.det_model.train()

    def train(self, mode=True):
        super().train(mode)
        if self.rgb_encoder.backbone_frozen:
            self.rgb_encoder.det_model.eval()

    def stop_parameters(self):
        return list(self.stop_net.parameters())


def supervised_loss(model, rgb, thermal, ultrasonic, labels, domain_meta=None):
    fused, _, _ = model.forward_logits(rgb, thermal, ultrasonic, domain_meta)
    return F.cross_entropy(fused, labels)


def compute_stop_rl_loss(model, value_net, rgb, thermal, ultrasonic, labels,
                         domain_meta=None, deterministic=False):
    """Single-step stop decision RL auxiliary loss on query samples."""
    device = rgb.device
    losses = []
    value_losses = []
    depths = []

    for i in range(rgb.size(0)):
        r = rgb[i:i + 1]
        t = thermal[i:i + 1]
        u = ultrasonic[i:i + 1]
        y = labels[i:i + 1]
        meta = domain_meta

        f_rgb, f_thermal, f_us = model.encode_modalities(r, t, u)
        fused, dynamic_w, _ = model.forward_logits(r, t, u, meta)
        state = model.build_state((f_rgb, f_thermal, f_us), fused, dynamic_w, 1.0)
        stop_logits = model.stop_logits(state)
        dist = Categorical(logits=stop_logits)
        action = stop_logits.argmax(dim=-1) if deterministic else dist.sample()

        correct = (fused.argmax(dim=-1) == y).float()
        conf = F.softmax(fused, dim=-1).max(dim=-1).values
        reward = correct + conf * 0.3 * correct - conf * 0.5 * (1.0 - correct)
        reward = reward.squeeze()

        baseline = value_net(state).squeeze(-1)
        advantage = reward - baseline.detach()
        losses.append(-dist.log_prob(action) * advantage)
        value_losses.append(F.mse_loss(baseline, reward.detach()))
        depths.append(1.0 if int(action.item()) == CLASSIFY else 2.0)

    policy_loss = torch.stack(losses).mean()
    value_loss = torch.stack(value_losses).mean()
    mean_depth = float(sum(depths) / max(len(depths), 1))
    return policy_loss + 0.5 * value_loss, mean_depth
