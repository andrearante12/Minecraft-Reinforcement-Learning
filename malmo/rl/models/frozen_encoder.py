"""
models/frozen_encoder.py
-------------------------
Frozen pretrained visual encoder for the vision world model
(VWM_INPUT="features"; see training/configs/hunting_vision_cfg.py).

Frames are encoded ONCE, in the trainer, by a pretrained backbone whose weights
never change. The world model consumes the resulting tokens, never pixels:

    frame (H,W,3) uint8 --[resize to VIS_RES, ImageNet norm]--> backbone
        --> CLS token (D) + patch tokens (P x P x D)
        --> patches avg-pooled to G x G (VIS_POOL_GRID)
        --> tokens (1 + G*G, D)   fp16, stored in the replay buffer

Backbones (ENCODERS):
  dinov2_vits14 — DINOv2 ViT-S/14 (21M params, D=384). Default. Weights are
                  fetched once via torch.hub (facebookresearch/dinov2), then cached.
  dinov2_vitb14 — DINOv2 ViT-B/14 (86M params, D=768).
  stub          — a fixed random patchify conv with the same token layout as
                  ViT-S/14. Offline tests only: no download, no meaning.

Not an nn.Module of the world model on purpose: it is never trained, never
checkpointed, and never enters an optimizer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODERS = {
    "dinov2_vits14": {"hub": ("facebookresearch/dinov2", "dinov2_vits14"), "dim": 384, "patch": 14},
    "dinov2_vitb14": {"hub": ("facebookresearch/dinov2", "dinov2_vitb14"), "dim": 768, "patch": 14},
    "stub":          {"hub": None, "dim": 384, "patch": 14},
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def token_shape(cfg):
    """(n_tokens, dim) produced for this config: CLS + G*G pooled patches."""
    g = cfg.VIS_POOL_GRID
    return 1 + g * g, ENCODERS[cfg.VIS_ENCODER]["dim"]


class _StubBackbone(nn.Module):
    """Deterministic random patch embedding mimicking DINOv2's forward_features."""

    def __init__(self, dim, patch):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=gen) * 0.05)
            self.proj.bias.zero_()
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x):
        p = self.norm(self.proj(x).flatten(2).transpose(1, 2))   # (N, P*P, D)
        return {"x_norm_clstoken": p.mean(dim=1), "x_norm_patchtokens": p}


class FrozenEncoder(nn.Module):
    def __init__(self, name="dinov2_vits14", pool_grid=4, res=224, device=None, half=True):
        super().__init__()
        if name not in ENCODERS:
            raise ValueError("Unknown VIS_ENCODER '{0}'; choose from {1}".format(name, sorted(ENCODERS)))
        spec = ENCODERS[name]
        if res % spec["patch"] != 0:
            raise ValueError("VIS_RES={0} must be a multiple of the patch size {1}".format(res, spec["patch"]))
        self.name, self.dim, self.patch = name, spec["dim"], spec["patch"]
        self.pool_grid, self.res = pool_grid, res
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"

        if spec["hub"] is None:
            backbone = _StubBackbone(self.dim, self.patch)
        else:
            backbone = torch.hub.load(*spec["hub"], verbose=False)
        self.backbone = backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.mean, self.std = self.mean.to(self.device), self.std.to(self.device)

    @property
    def n_tokens(self):
        return 1 + self.pool_grid * self.pool_grid

    def train(self, mode=True):
        # Always frozen: keep the backbone in eval mode whatever the caller does.
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames):
        """frames: (N,H,W,3) uint8 (numpy or tensor) -> (N, 1+G*G, D) fp16 tensor on device."""
        x = torch.as_tensor(np.asarray(frames) if not torch.is_tensor(frames) else frames,
                            device=self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-2:] != (self.res, self.res):
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear",
                              align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half):
            out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"].float()                         # (N, D)
        patches = out["x_norm_patchtokens"].float()                  # (N, P*P, D)
        n, p2, d = patches.shape
        side = int(round(p2 ** 0.5))
        grid = patches.transpose(1, 2).reshape(n, d, side, side)
        pooled = F.adaptive_avg_pool2d(grid, self.pool_grid).flatten(2).transpose(1, 2)
        return torch.cat([cls.unsqueeze(1), pooled], dim=1).half()

    def encode_np(self, frames, batch=64):
        """Chunked numpy -> numpy (N, T, D) float16. For demo/offline encoding."""
        outs = [self.forward(frames[i:i + batch]).cpu().numpy()
                for i in range(0, len(frames), batch)]
        if not outs:
            return np.zeros((0, self.n_tokens, self.dim), dtype=np.float16)
        return np.concatenate(outs, axis=0)


def build_frozen_encoder(cfg, device=None):
    return FrozenEncoder(cfg.VIS_ENCODER, cfg.VIS_POOL_GRID, getattr(cfg, "VIS_RES", 224), device)


def make_thumbs(frames, size=64):
    """(N,H,W,3) uint8 -> (N,size,size,3) uint8 area-downsampled thumbnails (CPU)."""
    x = torch.as_tensor(np.asarray(frames)).permute(0, 3, 1, 2).float()
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="area")
    return x.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
