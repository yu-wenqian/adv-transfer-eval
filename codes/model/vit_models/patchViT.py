import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import math
import random
from .utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import transforms

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

alpha = 1
beta = 1






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
        self.random_transforms = [
            # transforms.RandomResizedCrop(size = (self.patch_size, self.patch_size) ),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees= 30),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees= 30)
        ]
        self.transform_rate = 0.005

#https://discuss.pytorch.org/t/creating-non-overlapping-patches-and-reconstructing-image-back-from-the-patches/88417/2

    def patchify_and_merge(self, x):
        kc, kh, kw = x.shape[1], self.patch_size, self.patch_size  # kernel size
        dc, dh, dw = x.shape[1], self.patch_size, self.patch_size  # stride

        n_batches = x.shape[0]
        patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(n_batches, -1, kc, kh, kw)
        # print(patches.shape)
        # random select patches and permute
        with torch.no_grad():
            ind_patch = np.random.randint(low = 3,high=8,size= int(self.n_patches * self.transform_rate))
            for i in ind_patch:
                patches[:, i, :, :, :] = random.choice(self.random_transforms)(patches[:, i, :, :, :] )
            # x = random.choice(self.random_transforms)(x.clone())

        # Reshape back
        patches_orig = patches.view(unfold_shape)
        output_h = unfold_shape[2] * unfold_shape[4]
        output_w = unfold_shape[3] * unfold_shape[5]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(n_batches, kc, output_h, output_w)

        # Check for equality
        # print((patches_orig == x).all())
        return patches_orig

    def forward(self, x):
        x = self.patchify_and_merge(x)
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        # print(x.requires_grad)
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
                 drop_dim=None, drop_pos=None, drop_prob=None):
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
        self.drop_dim = drop_dim
        self.drop_pos = drop_pos
        self.drop_prob = drop_prob


    def forward(self, x, layer_ind = None):
        attn_out = drop(self.attn(self.norm1(x)), 1, self.drop_pos, self.drop_dim, self.drop_prob)
        x = drop(x, 0, self.drop_pos, self.drop_dim, self.drop_prob) + attn_out
        x = drop(x, 2, self.drop_pos, self.drop_dim, self.drop_prob)

        mlp_output = drop(self.mlp(self.norm2(x)), 4, self.drop_pos, self.drop_dim, self.drop_prob)
        x = drop(x, 3, self.drop_pos, self.drop_dim, self.drop_prob) + mlp_output
        x = drop(x, 5, self.drop_pos, self.drop_dim, self.drop_prob)

        return x


def drop(x, pos, drop_pos_list, dim, drop_prob):
    '''
    Args:
        pos: 0-5 current position
        drop_pos_list: corresponds to whether this position is dropped
        dim: drop dimension, list, len = 6 => separated drop
    Returns:
    '''
    if dim is None or not drop_pos_list[pos] or drop_prob == 0.:
        return x

    prob = drop_prob

    selected_dim = None

    if not len(dim) == 6:
        prob = prob / len(dim)
    elif len(dim) == 6:
        prob = prob / sum([x != -1 for x in dim])
        selected_dim = dim[pos]

    keep_prob = 1 - prob
    if (len(dim) != 6 and 0 in dim) or (len(dim) == 6 and selected_dim == 0):
        shape = x.shape  # drop neuron
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    if (len(dim) != 6 and 1 in dim) or (len(dim) == 6 and selected_dim == 1):
        shape = (x.shape[0],) + (x.shape[1],) + (1,) * (x.ndim - 2)  # #drop token
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    return x




class PatchViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., num_perm = 0,
                 token_combi = 0, drop_dim=None, drop_prob=None, drop_pos=None, min = 8, max = 20):

        super().__init__()

        self.drop_dim = drop_dim
        self.drop_pos = drop_pos
        self.drop_prob = drop_prob

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
                drop_dim=drop_dim, drop_pos=drop_pos, drop_prob=drop_prob
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

        widths = list(range(min, max+1))
        self.transforms = [transforms.Resize((x * 16, x * 16)) for x in widths]
        self.token_combi = token_combi
        self.num_perm = num_perm
        print(min, max)


    def forward(self, x, drop_pos=None, drop_dim=None, drop_prob=None):
        #trans = 0
        n_samples, nc, w, h = x.shape

        # interpolate token
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        #data augmentationa after concat
        pos_emb = self.pos_embed
        if '1' in self.token_combi:
            pos_emb = permute_2rows()
            # pos_emb = self.permute_2rows(int(self.num_perm * (w/224.)**2))
        if '0' in self.token_combi:
            pos_emb = interpolate_pos_encoding(self, _x, w, h)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + pos_emb
        _x = self.pos_drop(_x)


        for i, block in enumerate(self.blocks):
            _x = block(_x)
        _x = self.norm(_x)

        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        return _x


# load dataset => resize at runtime?
def load_patchViT(args):
    custom_config = load_config('vit_base_patch16_224')
    model = PatchViT(**custom_config, drop_prob = args.drop_prob, drop_dim=args.drop_dim, drop_pos = args.drop_pos,num_perm = args.num_perm,
                     token_combi = args.token_combi, min = args.min, max = args.max)
    model.eval()

    model_official = load_official_model('vit_base_patch16_224')
    print(type(model_official))

    assign_value(model_official, model)

    return model.to(args.device), model_official.to(args.device)
