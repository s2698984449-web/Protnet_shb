"""EFINet modules: ConditionEncoder + ProductEncoder + FiLMGate + DefectRepr.

Integrates with existing `multimodal_model.py` AdaptivePrototype (cross-attn).
Can be used standalone or plugged into the ProtoNet training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):
    """Encode a condition vector (one-hot or multi-hot) into a dense embedding.

    Args:
        n_conditions: number of condition factors (e.g. 6 from specimen_domain_map:
                      light / blur / contrast / noise / color / occlusion).
                      Pass raw scalar values or pre-binned one-hot vectors.
        d: output embedding dimension.  Defaults to 256.
    """

    def __init__(self, n_conditions: int, d: int = 256):
        super().__init__()
        self.n_conditions = n_conditions
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(n_conditions, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, n_conditions) or (B,) scalar -> one-hot."""
        if c.dim() == 1:
            c = F.one_hot(c.long(), self.n_conditions).float()
        return self.net(c)  # (B, d)


class ProductEncoder(nn.Module):
    """Encode product/component ID into a dense embedding.

    This is the "Component Encoder" from the EFINet architecture diagram.
    The product ID is a scalar integer (0..n_products-1) embedded like a word token.

    Args:
        n_products: total number of products across meta_train + meta_test.
                    meta_test product IDs are frozen (never seen in training) but
                    must be in the embedding table so the model can generalise.
        d: output embedding dimension.  Defaults to 256.
        freeze_meta_test: if True, meta_test product embeddings are frozen.
    """

    def __init__(self, n_products: int, d: int = 256, freeze_meta_test: bool = False):
        super().__init__()
        self.n_products = n_products
        self.d = d
        self.freeze_meta_test = freeze_meta_test
        self.emb = nn.Embedding(n_products, d)
        self._meta_test_mask: list[int] = []

    def set_meta_test_ids(self, meta_test_ids: list[int]):
        """Mark product IDs that appear only at meta-test time (frozen)."""
        self._meta_test_mask = meta_test_ids

    def forward(self, product_id: torch.Tensor) -> torch.Tensor:
        """product_id: (B,) long tensor of product IDs."""
        emb = self.emb(product_id)  # (B, d)
        if self.freeze_meta_test and self._meta_test_mask:
            for pid in self._meta_test_mask:
                mask = product_id == pid
                if mask.any():
                    emb = emb.clone()
                    emb[mask] = emb[mask].detach()
        return emb


class FiLMGate(nn.Module):
    """Feature-wise Linear Modulation (FiLM): condition scales and shifts backbone features.

    F' = (1 + gamma(c)) ⊙ F_img + beta(c)

    gamma, beta are predicted from the condition embedding.
    When c = zero vector, gamma=0, beta=0 → FiLM degenerates to identity (safe baseline).

    Args:
        d_cond:   dimension of condition embedding from ConditionEncoder.
        d_img:    number of channels in backbone features (C from backbone).
    """

    def __init__(self, d_cond: int, d_img: int):
        super().__init__()
        self.gamma = nn.Linear(d_cond, d_img)
        self.beta  = nn.Linear(d_cond, d_img)

    def forward(self, F_img: torch.Tensor, F_cond: torch.Tensor) -> torch.Tensor:
        """F_img: (B, C, H, W)  backbone spatial features.
           F_cond: (B, d_cond)  condition embedding."""
        g = self.gamma(F_cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        b = self.beta(F_cond).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return F_img * (1 + g) + b


class DefectRepr(nn.Module):
    """Fuse condition-aware backbone features + product embedding → defect representation.

    1. GAP on spatial features → (B, d_img)
    2. Concat with product embedding → (B, d_img + d_prod)
    3. MLP → (B, d_out) defect representation vector

    Args:
        d_img:   backbone channel count.
        d_prod:  product embedding dimension (must match ProductEncoder.d).
        d_out:   output defect representation dimension.  Defaults to 256.
    """

    def __init__(self, d_img: int, d_prod: int, d_out: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_img + d_prod, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, F_cond_aware: torch.Tensor, F_prod: torch.Tensor) -> torch.Tensor:
        """F_cond_aware: (B, C, H, W)  condition-modulated backbone features.
           F_prod:        (B, d_prod)  product embedding."""
        v = self.gap(F_cond_aware).flatten(1)          # (B, d_img)
        z = torch.cat([v, F_prod], dim=-1)              # (B, d_img + d_prod)
        return self.mlp(z)                              # (B, d_out)


class EFINetModel(nn.Module):
    """Standalone EFINet: YOLO26 backbone + Condition/Product encoding + FiLM + DefectRep + ProtoNet head.

    This is a self-contained model for few-shot evaluation.
    It exposes `.embed()` compatible with `protonet_loop.py` so you can drop it
    into the existing train.py / eval.py pipeline without changes.

    Usage:
        model = EFINetModel(n_conditions=6, n_products=15, product_ids=[...])
        model.freeze_backbone()
        # then pass to train.py / eval.py — model.embed() is called internally
    """

    def __init__(
        self,
        n_conditions: int = 6,
        n_products: int = 10,
        n_ways: int = 5,
        d_cond: int = 128,
        d_prod: int = 128,
        d_repr: int = 256,
        backbone_out_dim: int = 128,
        yolo_weights: str = '',
        yolo_imgsz: int = 640,
        use_multiscale_cff: bool = False,
        use_cbam: bool = False,
        freeze_meta_test_products: bool = True,
        meta_test_product_ids: list[int] | None = None,
    ):
        super().__init__()
        self.n_conditions = n_conditions
        self.n_products = n_products
        self.n_ways = n_ways
        self.d_repr = d_repr

        # ── Backbone ────────────────────────────────────────────────
        from .backbones import YOLO26BackboneEncoder
        self.backbone = YOLO26BackboneEncoder(
            out_dim=backbone_out_dim,
            imgsz=yolo_imgsz,
            weights_path=yolo_weights,
            use_multiscale_cff=use_multiscale_cff,
            use_cbam=use_cbam,
            fusion_embed_dim=backbone_out_dim,
        )
        self._backbone_out_dim = backbone_out_dim

        # ── Condition / Product / FiLM / Fusion ─────────────────────
        self.cond_enc = ConditionEncoder(n_conditions, d_cond)
        self.prod_enc = ProductEncoder(n_products, d_prod, freeze_meta_test=freeze_meta_test_products)
        if meta_test_product_ids:
            self.prod_enc.set_meta_test_ids(meta_test_product_ids)

        self.film = FiLMGate(d_cond, backbone_out_dim)
        self.defect_repr = DefectRepr(backbone_out_dim, d_prod, d_repr)

        # ── ProtoNet head (reuse existing AdaptivePrototype) ─────────
        from .protonet_loop import AdaptivePrototype, MahalanobisHead
        self.adaptive_proto = AdaptivePrototype(embed_dim=d_repr, num_heads=4)
        self.mahal_head = MahalanobisHead(feat_dim=d_repr, ways=n_ways)

        self._frozen = False

    def freeze_backbone(self):
        self.backbone.freeze_backbone()
        self._frozen = True

    def encode(self, rgb: torch.Tensor, condition: torch.Tensor, product_id: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to defect representations.

        Args:
            rgb:       (B, 3, H, W) RGB images.
            condition: (B, n_conditions) or (B,) scalar condition ID.
            product_id:(B,) long tensor of product IDs.

        Returns:
            (B, d_repr) defect representations.
        """
        F_img = self.backbone.encode(rgb)              # (B, C, H, W) — spatial features
        if self._frozen:
            F_img = F_img.detach()

        F_cond = self.cond_enc(condition)             # (B, d_cond)
        F_prod = self.prod_enc(product_id)             # (B, d_prod)
        F_cond_aware = self.film(F_img, F_cond)        # (B, C, H, W) — modulated
        Z = self.defect_repr(F_cond_aware, F_prod)    # (B, d_repr)
        return Z

    def embed(self, rgb, *args, **kwargs) -> torch.Tensor:
        """Duck-typed to match `multimodal_model.py AviationDefectModel.embed()`.

        For the standalone EFINet the extra args are ignored;
        condition + product_id should be set via `set_condition_product()` before
        the forward pass, or passed through domain_meta.

        Simplest usage in a ProtoNet loop that calls `model.embed(rgb)` directly:
            Set global state before each episode:
                model.set_episode_context(condition_batch, product_batch)
            and this embed() will read that context.
        """
        if hasattr(self, '_cond_batch') and hasattr(self, '_prod_batch'):
            return self.encode(rgb, self._cond_batch, self._prod_batch)
        # Fallback: zero condition + product 0 (safe but lossy)
        B = rgb.size(0)
        device = rgb.device
        n_cond = self.n_conditions
        n_prod = self.n_products
        cond_zeros = torch.zeros(B, n_cond, device=device)
        prod_zeros = torch.zeros(B, dtype=torch.long, device=device)
        return self.encode(rgb, cond_zeros, prod_zeros)

    def set_episode_context(self, condition: torch.Tensor, product_id: torch.Tensor):
        """Set per-episode condition + product vectors.

        Call this before each episode in the training/eval loop.
        Args:
            condition:  (B, n_conditions) or (B,) scalar.
            product_id: (B,) long — same product_id for all samples in an episode
                        (episode-level context).
        """
        self._cond_batch = condition
        self._prod_batch = product_id

    def meta_parameters(self):
        """All trainable parameters (for optimizer)."""
        return list(self.parameters())

    def __repr__(self):
        return (
            f"EFINetModel(cond_enc={self.n_conditions}d, "
            f"n_products={self.n_products}, d_repr={self.d_repr}, "
            f"backbone_out={self._backbone_out_dim})"
        )
