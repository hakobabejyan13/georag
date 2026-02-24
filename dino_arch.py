# dino_arch.py
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)             # (B,384,H/16,W/16)
        x = x.flatten(2)             # (B,384,N)
        x = x.transpose(1,2)         # (B,N,384)
        return x

class ViT_S16(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6):
        super().__init__()

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        return self.forward(x)
