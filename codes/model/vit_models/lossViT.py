import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import math
import random
import matplotlib.pyplot as plt

from torchvision import transforms

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

alpha = 1
beta = 1


def load_config(model):
    if model == "vit_base_patch16_224":
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
    elif model == "vit_small_patch16_224":
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
    elif model == "vit_large_patch16_224":
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


# visualize the attention for each head
# batch_size = 1
def plot_attn(attn):
    num_layers = len(attn)
    row, col, num_patch, _ = attn[0].size()
    # vmin, vmax = torch.min(attn), torch.max(attn)
    width = 3
    fig, axs = plt.subplots(nrows=num_layers, ncols=col, figsize=(col * width, num_layers * width))

    # #batch_size =1
    # if row == 1:
    #     j = 0
    #     for ax_col in axs:
    #         im = ax_col.imshow(attn[0][j].detach().numpy(), interpolation=None)
    #         j += 1
    # #batch_size > 1
    # else:
    for ax_row, i in zip(axs, range(num_layers)):
        j = 0
        for ax_col in ax_row:
            im = ax_col.imshow(attn[i][0][j].cpu().detach().numpy(), interpolation=None)
            # print(get_stats(grad[i][j]))
            j += 1
    plt.show()


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
        # print("Patch Emb.")
        # print(x.requires_grad)
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        # print(x.requires_grad)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Attention1(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

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

        # The only position to drop head????
        # if drop_dim == 1 or drop_dim == 2:
        #     attn = drop(attn, drop_dim, drop_prob ) #DROP HEAD, TOKEN

        return attn, v


class Attention2(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, v):
        attn = self.attn_drop(x)

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


class Block1(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,
                 drop_dim=None, drop_pos=None, drop_prob=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn1 = Attention1(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.attn2 = Attention2(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.drop_dim = drop_dim
        self.drop_pos = drop_pos
        self.drop_prob = drop_prob

    '''

    '''

    def forward(self, x):
        _attn, v = self.attn1(self.norm1(x))  # drop neuron, head, token head not here
        attn_out = drop(self.attn2(_attn, v), 1, self.drop_pos, self.drop_dim, self.drop_prob)
        x = drop(x, 0, self.drop_pos, self.drop_dim, self.drop_prob) + attn_out
        x = drop(x, 2, self.drop_pos, self.drop_dim, self.drop_prob)
        return x, _attn


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
        prob = drop_prob / len(dim)
    elif len(dim) == 6:
        prob = drop_prob / sum([x != -1 for x in dim])
        selected_dim = dim[pos]

    keep_prob = 1 - prob
    if (len(dim) != 6 and 0 in dim) or (len(dim) == 6 and selected_dim == 0):
        # if selected_dim == 0 or (dim is not None and 0 in dim):
        shape = x.shape  # drop neuron
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    if (len(dim) != 6 and 1 in dim) or (len(dim) == 6 and selected_dim == 1):
        shape = (x.shape[0],) + (x.shape[1],) + (1,) * (x.ndim - 2)  # #drop token
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    # if (len(dim) != 6 and 2 in dim ) or (len(dim) == 6 and selected_dim == 2):
    #     shape = (x.shape[0], 1,) + (x.shape[2],) + (1,) * (x.ndim -3)
    #     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    #     random_tensor.floor_()  # binarize
    #     x = x.div(keep_prob) * random_tensor
    ###TODO: if shape is same, overwrites...

    return x


class Block2(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,
                 drop_dim=None, drop_pos=None, drop_prob=None):
        super().__init__()
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

    def forward(self, x):
        """
        Returns torch.Tensor Shape `(n_samples, n_patches + 1, dim)`.
        """
        # TODO: BUG is here!!!
        mlp_output = drop(self.mlp(self.norm2(x)), 4, self.drop_pos, self.drop_dim, self.drop_prob)
        x = drop(x, 3, self.drop_pos, self.drop_dim, self.drop_prob) + mlp_output
        x = drop(x, 5, self.drop_pos, self.drop_dim, self.drop_prob)

        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768,
            depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.,
            drop_dim=None, drop_prob=None, drop_pos=None
    ):
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
            Block1(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                drop_dim=drop_dim, drop_pos=drop_pos, drop_prob=drop_prob
            )
            if i % 2 == 0
            else
            Block2(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                drop_dim=drop_dim, drop_pos=drop_pos, drop_prob=drop_prob
            )
            for i in range(2 * depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, loss_type=None):
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
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        _x = _x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)

        drop_index = None
        prob = self.drop_prob

        if isinstance(self.drop_dim, list):
            prob = self.drop_prob / len(self.drop_dim)

        if self.drop_dim == 3 or (self.drop_dim is not None and 3 in self.drop_dim):
            # drop half block
            rand = prob + np.random.rand(24)
            rand = np.floor(rand)
            drop_index = np.argwhere(rand > 0).flatten()

        if self.drop_dim == 4 or (self.drop_dim is not None and 4 in self.drop_dim):
            # drop block
            rand = prob + np.random.rand(12)
            rand = np.floor(rand)
            drop_ind = np.argwhere(rand > 0)
            drop_index = list(np.array([[2 * x, 2 * x + 1] for x in drop_ind]).flat)

        # self defined loss for all attn tokens
        for i, block in enumerate(self.blocks):
            if drop_index is not None:
                if i in drop_index:
                    continue

            if block.__class__.__name__ == 'Block1':
                _x, attn = block(_x)
                if loss_type is not None:
                    loss = get_loss(loss_type, attn)
                    self.loss.append(loss)

                # if n_samples != 1:
                #     plot_attn_map(attn, x, num_layer)
                #     del attn
            elif block.__class__.__name__ == 'Block2':
                _x = block(_x)
            # print(x.requires_grad)

        # attn: (batch_size, num_heads, 197, 197)

        # different loss function
        # self.loss.append(attn)
        _x = self.norm(_x)

        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        # return _x
        if loss_type == None:
            return _x
        else:
            return _x, self.loss


class PermutePatchViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., num_perm=2):
        super().__init__()
        self.num_perm = num_perm

    def select_one_patch(self, w, h):
        dim = self.pos_embed.shape[-1]
        N = self.pos_embed.shape[1] - 1
        x = random.randint(0, int(math.sqrt(N)) - w)
        y = random.randint(0, int(math.sqrt(N)) - h)

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch = patch_pos_embed[:, :, x:x + w, y:y + h]
        r = torch.randperm(w)
        c = torch.randperm(h)
        patch = patch[:, :, r, :]
        patch = patch[:, :, :, c]
        with torch.no_grad():
            patch_pos_embed[:, :, x:x + w, y:y + h] = patch

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def permute_rows(self):
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        N = self.pos_embed.shape[1] - 1
        perm = torch.randperm(N)
        patch_pos_embed = patch_pos_embed[:, perm, :]
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def permute_2rows(self, iter=1):
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        N = self.pos_embed.shape[1] - 1
        perm = list(range(N))
        for _ in range(iter):
            swap_list = random.sample(range(N), 2)
            perm[swap_list[0]], perm[swap_list[1]] = perm[swap_list[1]], perm[swap_list[0]]
        patch_pos_embed = patch_pos_embed[:, perm, :]
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def permute_pos_emb(self):
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]
        N = self.pos_embed.shape[1] - 1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        a, b = patch_pos_embed.split(7, dim=2)
        a1, a2 = a.split(7, dim=-1)
        b1, b2 = b.split(7, dim=-1)

        def shuffle(a):
            a = a[:, :, :, torch.randperm(a.shape[-1])]
            a = a[:, :, torch.randperm(a.shape[-2]), :]
            return a

        reconstructed = torch.cat([torch.cat([shuffle(a1), shuffle(a2)], dim=-1),
                                   torch.cat([shuffle(b1), shuffle(b2)], dim=-1)], dim=2)
        patch_pos_embed = reconstructed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, loss_type=None, drop_pos=None, drop_dim=None, drop_prob=None):
        n_samples, nc, w, h = x.shape

        # interpolate token
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        # _x = _x + self.pos_embed
        _x = _x + self.permute_2rows(
            self.num_perm)  # (n_samples, 1 + n_patches, embed_dim) #different permutation for each batch?
        _x = self.pos_drop(_x)

        for i, block in enumerate(self.blocks):
            if block.__class__.__name__ == 'Block1':
                _x, attn = block(_x)
            elif block.__class__.__name__ == 'Block2':
                _x = block(_x)
        _x = self.norm(_x)

        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        if loss_type == None:
            return _x
        else:
            return _x, self.loss


class VarInputLenViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., min=12, max=16):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        widths = list(range(min, max + 1))
        self.transforms = [transforms.Resize((x * 16, x * 16)) for x in widths]
        print(f'min: {min}, max: {max}')

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1  # ?
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = self.pos_embed.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, loss_type=None, drop_pos=None, drop_dim=None, drop_prob=None):
        x = random.choice(self.transforms)(x.clone())
        n_samples, nc, w, h = x.shape

        # interpolate token
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + self.interpolate_pos_encoding(_x, w, h)  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)

        for i, block in enumerate(self.blocks):
            if block.__class__.__name__ == 'Block1':
                _x, attn = block(_x)
            elif block.__class__.__name__ == 'Block2':
                _x = block(_x)
        _x = self.norm(_x)

        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        if loss_type == None:
            return _x
        else:
            return _x, self.loss


class DropViT(VisionTransformer):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., scale=(0.08, 1), ratio=(3. / 4., 4. / 3.),
                 ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block1(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p, )
            if i % 2 == 0
            else
            Block2(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p, )
            for i in range(2 * depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.ratio = ratio
        self.scale = scale

    def forward(self, x, loss_type=None, drop_dim=None, drop_prob=None):
        _, _, img_size, _ = x.shape
        randomResizeCrop = transforms.RandomResizedCrop((img_size, img_size), scale=self.scale, ratio=self.ratio)

        outputs = []
        for i in range(3):
            random_x = randomResizeCrop(x.clone())
            randomized_output = self.forward_step(random_x)
            outputs.append(randomized_output)
        return outputs

    def forward_step(self, x, loss_type=None, drop_dim=None, drop_prob=None):
        self.loss = []
        n_samples = x.shape[0]
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim) #TODO: dimension mismatch
        _x = self.pos_drop(_x)
        # print(x.requires_grad)
        drop_index = None
        if drop_dim == 3:
            # drop half block
            rand = drop_prob + np.random.rand(24)
            rand = np.floor(rand)
            drop_index = np.argwhere(rand > 0).flatten()

        if drop_dim == 4:
            # drop block
            rand = drop_prob + np.random.rand(12)
            rand = np.floor(rand)
            drop_ind = np.argwhere(rand > 0)
            drop_index = list(np.array([[2 * x, 2 * x + 1] for x in drop_ind]).flat)

        num_layer = 0
        # self defined loss for all attn tokens
        for i, block in enumerate(self.blocks):
            if drop_index is not None:
                if i in drop_index:
                    continue

            if block.__class__.__name__ == 'Block1':
                _x, attn = block(_x, drop_dim, drop_prob)
                if loss_type is not None:
                    loss = get_loss(loss_type, attn)
                    self.loss.append(loss)

                # if n_samples != 1:
                #     plot_attn_map(attn, x, num_layer)
                #     del attn
            elif block.__class__.__name__ == 'Block2':
                _x = block(_x)

            num_layer += 1
            # print(x.requires_grad)

        # attn: (batch_size, num_heads, 197, 197)

        # different loss function
        # self.loss.append(attn)
        _x = self.norm(_x)

        cls_token_final = _x[:, 0]  # just the CLS token
        _x = self.head(cls_token_final)
        # return _x
        if loss_type == None:
            return _x
        else:
            return _x, self.loss


def load_multiscale_vit(device, args):
    custom_config = load_config('vit_base_patch16_224')
    model = MultiScaleViT(**custom_config, scale=args.scale, ratio=args.ratio)
    model.eval()

    model_official = load_official_model('vit_base_patch16_224')
    print(type(model_official))

    for (n_o, p_o), (n_c, p_c) in zip(
            model_official.named_parameters(), model.named_parameters()
    ):
        # print(n_o, n_c)
        assert p_o.numel() == p_c.numel()
        # print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        assert_tensors_equal(p_c.data, p_o.data)

    # inp = torch.rand(1, 3, 224, 224)
    # res_c = model(inp)
    # res_o = model_official(inp)
    return model.to(device), model_official.to(device)


def plot_attn_map(attn, x, num_layer):
    _, channel, width, height = x.shape
    batch_size, num_head, dim, _ = attn.shape

    attn = attn[:, :, 0, 1:].view((batch_size, num_head, 14, 14))
    up_func = nn.Upsample(size=(width, height), mode='nearest')
    # up_func = nn.Upsample(scale_factor=( int(width / dim), int(height / dim)), mode='nearest')
    # attn = up_func(attn)
    attn = up_func(attn)
    row = batch_size
    col = num_head

    # vmin, vmax = torch.min(attn), torch.max(attn)
    width = 6
    fig, axs = plt.subplots(nrows=row, ncols=col + 1, figsize=(col * width, row * width))

    # #batch_size =1

    for ax_row, i in zip(axs, range(row)):
        j = 0
        img = x[i].permute(1, 2, 0).cpu().detach().numpy()
        for ax_col in ax_row:
            if j == 0:
                ax_col.imshow(img)
            else:
                ax_col.imshow(img, alpha=0.9)
                ax_col.imshow(attn[i][j - 1].cpu().detach().numpy(), interpolation=None, alpha=0.5)
            # print(get_stats(grad[i][j]))
            j += 1
    plt.savefig(f'figs/layer_{num_layer}.png')
    plt.show()


def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)


def load_official_model(model_name):
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    # model_official.to(device)
    return model_official


def load_DropViT(device, args):
    custom_config = load_config(args.src_model)
    model = VisionTransformer(**custom_config, drop_pos=args.drop_pos, drop_prob=args.drop_prob,
                              drop_dim=args.drop_dim, token_combi=args.token_combi)
    model.eval()

    src_model_name = 'vit' + args.src_model.split('vit')[-1]
    model_official = load_official_model(src_model_name)
    print(type(model_official))
    assign_value(model_official, model)

    # Asserts
    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)
    # Asserts
    if args.drop_prob == 0:
        assert_tensors_equal(res_c, res_o)
    # del model_official
    assert get_n_params(model) == get_n_params(model_official)

    return model.to(device), model_official.to(device)


def load_vit(device, model_name):
    custom_config = load_config(model_name)
    model = VisionTransformer(**custom_config)
    model.eval()

    model_official = load_official_model(model_name)
    print(type(model_official))

    for (n_o, p_o), (n_c, p_c) in zip(
            model_official.named_parameters(), model.named_parameters()
    ):
        # print(n_o, n_c)
        assert p_o.numel() == p_c.numel()
        # print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        assert_tensors_equal(p_c.data, p_o.data)

    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)

    if model_name == "vit_large_patch16_224":
        res_o = res_o[-1]  # TODO output doesn't match

    # Asserts
    assert get_n_params(model) == get_n_params(model_official)
    # assert_tensors_equal(res_c, res_o)
    # del model_official
    if model_name == 'vit_large_patch16_224':
        return model_official.to(device)
    return model.to(device), model_official.to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def load_permute_vit(device, args):
    custom_config = load_config('vit_base_patch16_224')
    model = PermutePatchViT(**custom_config, num_perm=args.num_perm)
    model.eval()

    model_official = load_official_model('vit_base_patch16_224')
    print(type(model_official))

    assign_value(model_official, model)

    inp = torch.rand(1, 3, 224, 224)
    res_c = model(inp)
    res_o = model_official(inp)

    # Asserts
    # assert_tensors_equal(res_c, res_o)
    # del model_official
    assert get_n_params(model) == get_n_params(model_official)
    return model.to(device), model_official.to(device)


def assign_value(modelA, modelB):
    for (n_o, p_o), (n_c, p_c) in zip(
            modelA.named_parameters(), modelB.named_parameters()
    ):
        # print(n_o, n_c)
        assert p_o.numel() == p_c.numel()
        # print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        assert_tensors_equal(p_c.data, p_o.data)


# load dataset => resize at runtime?
def load_varinput_vit(device, args):
    custom_config = load_config('vit_base_patch16_224')
    model = VarInputLenViT(**custom_config, min=args.min, max=args.max)
    model.eval()

    model_official = load_official_model('vit_base_patch16_224')
    print(type(model_official))

    assign_value(model_official, model)

    return model.to(device), model_official.to(device)


def load_reduce_vit(device, args):
    model = load_vit(device, "vit_base_patch16_224")

    if args.layer_index is not None:
        layer_index = [args.layer_index]
        if args.layer == 'mlp':
            layer_index = [2 * x + 1 for x in layer_index]
        elif args.layer == 'mh':
            layer_index = [2 * x for x in layer_index]
        elif args.layer == 'both':
            layer_index = sum([[2 * x, 2 * x + 1] for x in layer_index], [])
        print(layer_index)

        for i in layer_index:
            model.blocks[i] = Identity()

    return model.to(device)


def load_vit_with_specific_layers(device, model, reduce_layer=[0, 5, 11]):
    model = load_vit(device, "vit_base_patch16_224")

    layer_index = sum([[2 * x, 2 * x + 1] for x in reduce_layer], [])
    print(layer_index)
    model.blocks = nn.ModuleList([model.blocks[i] for i in layer_index])
    return model.to(device)


def load_vit_skip_res(device, args):
    model = load_vit(device, "vit_base_patch16_224")

    # register hook
    global beta
    global alpha
    beta = args.beta
    alpha = args.alpha
    register_skip_gradient_vit(model)
    return model.to(device)



