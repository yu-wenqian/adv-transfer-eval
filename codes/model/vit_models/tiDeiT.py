import torch
import torch.nn as nn
from torchvision import transforms
import timm
import os
import random
from timm.models.layers import trunc_normal_

from .utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_config(model):
    if model == "idvit_base_patch16_224":
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif model == "idvit_small_patch16_224":
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 8,
            "n_heads": 8,
            "qkv_bias": False,
            "mlp_ratio": 3,
        }
    elif model == "idvit_large_patch16_224":
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "n_heads": 16,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }



class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)



    # def forward(self, x, drop_dim = None, drop_prob = None):
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.,
                 drop_dim=None, drop_pos=None, drop_prob=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop_dim = drop_dim
        self.drop_pos = drop_pos
        self.drop_prob = drop_prob
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )


    def forward(self, x ):
        x = x + self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x)) + x
        return x


class TIDeiT(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768,
            depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.img_size = img_size

        self.embed_dim = embed_dim
        self.num_classes = n_classes


        self.training = False



    def random_transform(self, x):
        if  torch.rand(()) < 0.5:
            return x

        rnd = torch.randint(int(self.img_size * 0.9), self.img_size, ())
        h_rem = self.img_size - rnd
        w_rem = self.img_size - rnd

        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, h_rem, ())
        pad_right = w_rem - pad_left
        resize_pad = transforms.Compose([transforms.Resize((rnd, rnd)),
                                        transforms.Pad(padding = [pad_left, pad_top, pad_right, pad_bottom])])
        return resize_pad(x.clone())

    def random_shift(self, x):
        shift_range = 10. /self.img_size
        shift = transforms.RandomAffine(0, translate = (shift_range, shift_range))
        return shift(x.clone())

    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor Logits over all the classes - `(n_samples, n_classes)`.
        """
        self.loss = []
        n_samples = x.shape[0]
        # trasnform x
        x = self.random_transform(x)
        x = self.random_shift(x)

        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        # dist_token = self.dist_token.expand(n_samples, -1, -1) ##

        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)

        # self defined loss for all attn tokens
        for i, block in enumerate(self.blocks):
            _x = block(_x)

        _x = self.norm(_x)
        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        return _x

    # def forward(self, x):
    #     x, x_dist = self.forward_features(x)
    #     x = self.head(x)
    #     x_dist = self.head_dist(x_dist)
    #     if self.training:
    #         return x, x_dist
    #     else:
    #         # during inference, return the average of both classifier predictions
    #         return (x + x_dist) / 2


def load_tiDeiT(model_name, args):
    custom_config = load_deit_config(model_name)
    model = TIDeiT(**custom_config)
    model.eval()

    model_official = load_deit_official(model_name[2:])
    print(type(model_official))

    assign_value(model_official, model)

    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)

    assert get_n_params(model) == get_n_params(model_official)
    # assert_tensors_equal(res_c, res_o)

    return model.to(args.device), model_official.to(args.device)




