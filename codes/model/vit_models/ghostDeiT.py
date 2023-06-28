import torch
import torch.nn as nn
import timm
import os

from .utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from timm.models.layers import trunc_normal_



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


    def forward(self, x, skip_eros = 0, drop_layer = 0 ):
        x = x * skip_eros + self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x)) + x
        x = drop(x, drop_layer)
        return x


def drop(x, prob):
    if prob == 0.:
        return x

    keep_prob = 1 - prob
    # if selected_dim == 0 or (dim is not None and 0 in dim):
    shape = x.shape  # drop neuron
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    x = x.div(keep_prob) * random_tensor

    return x


class GhostDeiT(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768,
            depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.,
             prob_skip_eros = 0.01, prob_drop_layer = 0.01
    ):
        # super().__init__( img_size=img_size, patch_size=patch_size, in_chans=in_chans, n_classes=n_classes,
        #          embed_dim=embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,)
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+ self.patch_embed.n_patches, embed_dim)
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

        self.prob_skip_eros = prob_skip_eros
        self.prob_drop_layer = prob_drop_layer
        self.depth = depth

        self.embed_dim = embed_dim
        self.num_classes = n_classes

        # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        # self.head_dist.apply(self._init_weights)
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.dist_token, std=.02)

        self.training = False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor Logits over all the classes - `(n_samples, n_classes)`.
        """
        #resample in each iteration
        self.skip_eros = ( np.random.rand(self.depth)*2 - 1 ) * self.prob_skip_eros + 1
        self.drop_layer = np.random.rand(self.depth) * self.prob_drop_layer

        self.loss = []
        n_samples = x.shape[0]
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        # dist_token = self.dist_token.expand(n_samples, -1, -1) ##

        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)

        # self defined loss for all attn tokens
        for i, block in enumerate(self.blocks):
            _x = block(_x, self.skip_eros[i], self.drop_layer[i])

        # _x = self.norm(_x)
        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        # return _x[:, 0], _x[:, 1]
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
    #


def load_ghostDeiT(model_name, args):
    custom_config = load_deit_config(model_name)
    model = GhostDeiT(**custom_config,  prob_skip_eros = args.skip_eros,
                     prob_drop_layer = args.drop_layer)
    model.eval()

    model_official = load_deit_official(model_name[5:])
    print(type(model_official))

    assign_value(model_official, model)

    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)

    assert get_n_params(model) == get_n_params(model_official)
    # assert_tensors_equal(res_c, res_o)

    return model.to(args.device), model_official.to(args.device)




