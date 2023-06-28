# SIM variant: no for loop but multiple scales at last layer

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import os
import random

from .utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




#vit with zero padding
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


class MSViT(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768,
            depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., min = 8, max = 20, args = None
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
        self.range = list(np.arange(self.img_size * 0.8, self.img_size * 1.6, self.img_size* 0.1))
        self.layer = args.layer
        self.mode = args.mode
        self.kernelrange = args.kernelrange
        print(self.layer, self.mode, self.kernelrange)

        widths = list(range(min, max+1))
        self.transforms = [transforms.Resize((x * 16, x * 16)) for x in widths]
        self.num_resize = args.num_resize


    def rescale_1d(self, x, kernel_size = 2, mode = 'avg'):
        class_token = torch.unsqueeze(x[:, 0, :], 1)
        if mode == 'avg':
            pooling = torch.nn.AvgPool1d(kernel_size=kernel_size, stride= kernel_size)
        elif mode == 'max':
            pooling = torch.nn.MaxPool1d(kernel_size=kernel_size, stride= kernel_size)
        output = pooling(x[:, 1:, :].transpose(1, 2))
        return torch.cat((class_token, output.transpose(1, 2)), dim = 1)


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

        # trasnform x
        x = random.choice(self.transforms)(x.clone())
        n_samples, nc, w, h = x.shape

        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        pos_emb = interpolate_pos_encoding(self, _x, w, h)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + pos_emb  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)
        outputs = []
        # self defined loss for all attn tokens
        for i, block in enumerate(self.blocks):
            if i in self.layer:
                outputs = []
                for _ in range(self.num_resize):
                    rand_kernel_size = np.random.randint(self.kernelrange[0], self.kernelrange[1])
                    rand_mode = np.random.choice(self.mode)
                    rescaled_x = self.rescale_1d(_x.clone(),  kernel_size= rand_kernel_size, mode = rand_mode)
                    output = block(rescaled_x)

                    output = self.norm(output)
                    cls_token_final = output[:, 0]  # just the CLS token
                    output = self.head(cls_token_final)

                    outputs.append(output)
                return outputs
            else:
                _x = block(_x)

        return outputs


def load_msViT(args):
    custom_config = load_config(args.src_model)
    model = MSViT(**custom_config, args = args)
    model.eval()

    model_official = load_official_model(args.src_model[2:])
    print(type(model_official))

    assign_value(model_official, model)

    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)

    assert get_n_params(model) == get_n_params(model_official)
    # assert_tensors_equal(res_c, res_o)

    return model.to(args.device), model_official.to(args.device)


