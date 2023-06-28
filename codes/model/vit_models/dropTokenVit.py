import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

from .dropViT import load_official_model, assign_value
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






class DropTokenViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., drop_prob=0):
        super().__init__()
        self.drop_prob = drop_prob

    def dropToken(self, x):
        if self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (x.shape[1],) + (1,) * (x.ndim - 2)  # #drop token
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
        drop_positions = random

    def generate_drop_index(self, x): #TODO:
        n = x.shape[1] - 1
        rand_tensor = 1 - self.drop_prob + torch.rand((n,), device=x.device, dtype=x.dtype)
        rand_tensor.floor_()
        rand_tensor = torch.cat((torch.tensor([1], device=x.device,), rand_tensor))
        self.drop_ind = torch.nonzero(rand_tensor)

    def forward(self, x, loss_type=None, drop_pos=None, drop_dim=None, drop_prob=None):
        n_samples, nc, w, h = x.shape

        # interpolate token
        _x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        _x = torch.cat((cls_token, _x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        _x = _x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        _x = self.pos_drop(_x)

        for i, block in enumerate(self.blocks):
            # drop tokens here
            self.generate_drop_index(_x)
            _x = _x[:, self.drop_ind, :].squeeze()

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






class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



# load dataset => resize at runtime?
def load_droptoken_vit(device, args):
    custom_config = load_config('vit_base_patch16_224')
    model = DropTokenViT(**custom_config, drop_prob= args.drop_prob)
    model.eval()

    model_official = load_official_model('vit_base_patch16_224')
    print(type(model_official))

    assign_value(model_official, model)

    return model.to(device), model_official.to(device)

