"""Vision Transformer (ViT) implementation for BUSI classification.

A compact ViT following "An Image is Worth 16x16 Words" (Dosovitskiy et al.,
2020), using a learnable class token and a patch embedding.
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split an image into patches and linearly project each patch."""

    def __init__(self, in_channels, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # (B, dim, H/P, W/P)
        x = x.flatten(2)              # (B, dim, N)
        x = x.transpose(1, 2)         # (B, N, dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 num_classes=3, dim=256, depth=6, heads=8, mlp_dim=512,
                 dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert dim % heads == 0, "dim must be divisible by heads"

        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]  # class token
        x = self.head(x)
        return x


def build_vit(image_size=224, patch_size=16, num_classes=3, dim=256,
              depth=6, heads=8, mlp_dim=512, dropout=0.1, **kwargs):
    return ViT(image_size=image_size, patch_size=patch_size,
               num_classes=num_classes, dim=dim, depth=depth, heads=heads,
               mlp_dim=mlp_dim, dropout=dropout, **kwargs)


def build_pretrained_vit(variant="b16", num_classes=3, image_size=224):
    """Build a torchvision ViT initialized with ImageNet weights.

    The classification head is replaced with a fresh ``num_classes``-way
    layer (torchvision forces 1000 classes when pretrained weights are
    loaded, so the head is swapped afterwards).
    """
    from torchvision import models as tv_models

    variants = {
        "b16": (tv_models.vit_b_16, tv_models.ViT_B_16_Weights.IMAGENET1K_V1),
        "b32": (tv_models.vit_b_32, tv_models.ViT_B_32_Weights.IMAGENET1K_V1),
        "l16": (tv_models.vit_l_16, tv_models.ViT_L_16_Weights.IMAGENET1K_V1),
        "l32": (tv_models.vit_l_32, tv_models.ViT_L_32_Weights.IMAGENET1K_V1),
    }
    if variant not in variants:
        raise ValueError(f"Unknown ViT variant: {variant} "
                         f"(choose from {sorted(variants)})")

    builder, weights = variants[variant]
    model = builder(weights=weights, image_size=image_size)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
