# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from .utils import *
import random
import timm


def load_deit(device, model_name):
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    return model_official.to(device)

def load_deit_config(model):
    #     patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if "deit_tiny_distilled_patch16_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 192,
            "depth": 12,
            "n_heads": 3,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    # patch_size = 16, embed_dim = 384, depth = 12, num_heads = 6, mlp_ratio = 4, qkv_bias = True,
    # norm_layer = partial(nn.LayerNorm, eps=1e-6), ** kwargs)
    elif "deit_small_distilled_patch16_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
            "qkv_bias": True,
            "mlp_ratio": 3,
        }
    # patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = True,
    # norm_layer = partial(nn.LayerNorm, eps=1e-6), ** kwargs)
    elif "deit_base_distilled_patch16_224" in model:
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

#https://github.com/facebookresearch/deit/blob/main/models.py
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
# load dataset => resize at runtime?
def load_deit(device, args):
    custom_config = load_deit_config('deit_tiny_distilled_patch16_224')
    model = DistilledVisionTransformer(**custom_config, drop_prob = args.drop_prob, drop_dim=args.drop_dim, drop_pos = args.drop_pos,num_perm = args.num_perm,
                     token_combi = args.token_combi, min = args.min, max = args.max)
    model.eval()

    model_official = load_official_model('deit_tiny_distilled_patch16_224')
    print(type(model_official))

    assign_value(model_official, model)

    return model.to(device), model_official.to(device)