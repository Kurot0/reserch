import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x):
        return self.proj(x)


class PatchExpansion(nn.Module):
    def __init__(self, embed_dim, patch_size, out_chans, img_size_in):
        super().__init__()
        self.proj_transpose = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange("b (h w) c -> b c h w", h=img_size_in[0] // patch_size[0]),
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_size)
        )
    
    def forward(self, x):
        return self.proj_transpose(x)


# class PatchExpansion(nn.Module):
#     def __init__(self, embed_dim, patch_size, out_chans, img_size_in):
#         super().__init__()
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)
#         self.rearrange = Rearrange(
#             "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
#             p1=patch_size[0],
#             p2=patch_size[1],
#             h=img_size_in[0] // patch_size[0],
#         )

#     def forward(self, x):
#         x = self.norm(x)
#         x = self.head(x)
#         x = self.rearrange(x)
#         return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.model(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=0.):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b p c -> b c p"),
            Mlp(num_patches, tokens_mlp_dim, drop=drop),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, channels_mlp_dim, drop=drop)
        )
    
    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class Network(nn.Module):
    def __init__(self, img_size_in, img_size_out, patch_size, in_chans, out_chans, embed_dim, 
                 tokens_mlp_dim, channels_mlp_dim, num_blocks, drop_rate=0., **kwargs):
        super().__init__()
        num_patches = (img_size_in[1] // patch_size[1]) * (img_size_in[0] // patch_size[0])
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        layers = [ 
            MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
            for _ in range(num_blocks)]
        self.mixer_layers = nn.Sequential(*layers)
        self.patch_expand = PatchExpansion(embed_dim, patch_size, out_chans, img_size_in)
        self.downsample = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    def forward(self, x, underground_data):
        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.patch_expand(x)
        x = self.downsample(x)
        return x
