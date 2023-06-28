import torch
import numpy as np
import timm
import random
import math
import torch.nn as nn
from functools import partial

def permute_2rows(self):
    if self.num_perm == 0:
        return self.pos_embed
    class_pos_embed = self.pos_embed[:, 0]
    patch_pos_embed = self.pos_embed[:, 1:]
    N = self.pos_embed.shape[1] - 1
    perm = list(range(N))
    for _ in range(self.num_perm):
        swap_list = random.sample(range(N), 2)
        perm[swap_list[0]], perm[swap_list[1]] = perm[swap_list[1]], perm[swap_list[0]]
    patch_pos_embed = patch_pos_embed[:, perm, :]
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

def interpolate_pos_encoding_distilled_deit(self, x, w, h):
    npatch = x.shape[1] - 1  # ?
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    class_pos_embed = self.pos_embed[:, 0]
    dist_pos_emb = self.pos_embed[:, 1]
    patch_pos_embed = self.pos_embed[:, 2:]
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
    return torch.cat((class_pos_embed.unsqueeze(0), dist_pos_emb.unsqueeze(1),
                      patch_pos_embed), dim=1)



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


def load_config(model):
    if "vit_base_patch16_224" in model:
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
    elif "vit_base_patch32_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 32,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_base_patch8_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 8,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_small_patch16_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_small_patch32_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 32,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_small_patch8_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 8,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_large_patch16_224" in model:
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
    elif "vit_large_patch32_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 32,
            "embed_dim": 1024,
            "depth": 24,
            "n_heads": 16,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }
    elif "vit_large_patch14_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 14,
            "embed_dim": 1024,
            "depth": 24,
            "n_heads": 16,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }


def load_deit_config(model):
    # model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    if "deit_tiny_patch16_224" in model:
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
    # model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    elif "deit_small_patch16_224" in model:
        return {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "n_heads": 6,
            "qkv_bias": True,
            "mlp_ratio": 4,
        }

        # model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    elif "deit_base_patch16_224" in model:
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

def search_model_name(model_name):
    if "vit_base_patch16_224" in model_name:
        return 'vit_base_patch16_224'
    elif "vit_base_patch32_224" in model_name:
        return 'vit_base_patch32_224'
    elif "vit_base_patch8_224" in model_name:
        return 'vit_base_patch8_224'
    elif "vit_small_patch16_224" in model_name:
        return 'vit_small_patch16_224'
    elif "vit_small_patch32_224" in model_name:
        return 'vit_small_patch32_224'
    elif "vit_small_patch8_224" in model_name:
        return 'vit_small_patch8_224'
    elif "vit_large_patch16_224" in model_name:
        return 'vit_large_patch16_224'
    elif "vit_large_patch32_224" in model_name:
        return 'vit_large_patch32_224'
    elif "vit_large_patch14_224" in model_name:
        return 'vit_large_patch14_224'

def search_deit_model_name(model_name):
    if 'deit_small_patch16_224' in model_name:
        return 'deit_small_patch16_224'
    elif 'deit_base_patch16_224' in model_name:
        return 'deit_base_patch16_224'
    elif 'deit_tiny_patch16_224' in model_name:
        return 'deit_tiny_patch16_224'

def load_official_model(model_name):
    model_name = search_model_name(model_name)
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    return model_official

def load_deit_official(model_name):
    model_name = search_deit_model_name(model_name)
    model_official = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
    model_official.eval()
    model_official.training = False
    return model_official


def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    np.testing.assert_allclose(a1, a2)

def assign_value(modelA, modelB):
    for (n_o, p_o), (n_c, p_c) in zip(
            modelA.named_parameters(), modelB.named_parameters()
    ):
        # print(n_o, n_c,  p_o.numel(), p_c.numel())
        # assert p_o.numel() == p_c.numel()
        # print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data
        # assert_tensors_equal(p_c.data, p_o.data)




def backward_hook_vit(module, grad_input, grad_out):
    # print("module hook: ", module)
    # print('grad_input', grad_input[0].size(), grad_input[1].size())
    # print(torch.max(grad_input[0]), torch.min(grad_input[0]))
    # print(torch.max(grad_input[1]), torch.min(grad_input[1]))
    # print('grad_out', grad_out[0].size())
    global alpha
    global beta
    # gamma = np.power(gamma, 0.5)
    # beta = module.beta
    # alpha = module.alpha
    # print(f"beta: {beta}, alpha: {alpha}")
    return (grad_input[0] * beta, grad_input[1] * alpha)


def register_skip_gradient_vit(model):
    for name, module in model.named_modules():
        if (len(name) == 8 or len(name) == 9) and 'blocks' in name:
            # print('Name: ',name)
            # print('Module: ',module)
            module.register_backward_hook(backward_hook_vit)
            # module.register_backward_hook(backward_hook_norm)

def register_hook_for_vit(model,arch):
    def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
        mask = torch.ones_like(grad_in[0]) * gamma
        return (mask * grad_in[0][:], )

    drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

    if arch in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
        for i in range(12):
            model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)

def get_loss(loss_type, attn ):
    attn_map = attn[:, :, 0, 1:]
    if loss_type == 'std_attn':
        loss = torch.std(attn_map, -1).sum()
    elif loss_type == 'one_minus':
        ones = torch.ones_like(attn_map)
        criterion = torch.nn.MSELoss()
        loss = criterion(attn_map, ones - attn_map)
    elif loss_type == 'inverted_sign':
        criterion = torch.nn.MSELoss()
        loss = criterion(attn_map, -attn_map)
    elif loss_type == 'func':
        batch_size, num_head, _ = attn_map.size()
        targ = torch.ones_like(attn_map) * torch.max(attn_map) - attn_map
        norm_targ = targ / torch.sum(targ, dim=-1).view(batch_size, num_head, 1)
        criterion = torch.nn.MSELoss()
        loss = criterion(norm_targ, attn_map)
    elif loss_type == 'patch_to_class':
        pass
    elif loss_type == 'all_tokens':
        criterion = torch.nn.MSELoss()
        batch_size, num_head, _ , _= attn.size()
        targ = torch.ones_like(attn) * torch.max(attn) - attn
        norm_targ = targ / torch.sum(targ, dim=-1).view(batch_size, num_head, 197, 1)
        loss = criterion(norm_targ, attn)

    return loss