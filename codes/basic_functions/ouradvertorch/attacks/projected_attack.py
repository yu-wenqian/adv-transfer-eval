from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import numpy as np
import torch
import torch.nn as nn

from random import randint

from torchvision import transforms
from PIL import Image

from ..utils import clamp, normalize_by_pnorm, rand_init_delta
from .interaction_loss import (InteractionLoss, get_features,
                               sample_for_interaction)
from codes.utils.util_linbp import linbp_forw_resnet50, linbp_backw_resnet50,ila_forw_resnet50,ILAProjLoss
from codes.utils.util_ila import get_source_layers,ILA,model_configs
from codes.model.load_model import load_ila_models

from codes.model.gfnet.inference import eval_gfnet_ensemble
from codes.model.gfnet.utils import get_prime

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def transition_invariant_conv(size=15):
    kernel = gkern(size, 3).astype(np.float32)
    padding = size // 2
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=size,
        stride=1,
        groups=3,
        padding=padding,
        bias=False)
    conv.weight.data = conv.weight.new_tensor(data=stack_kernel)

    return conv


def input_diversity(input_tensor,image_width,image_resize,prob):
    if prob > 0.0:
        rnd = randint(image_width,image_resize)
        rescaled = transforms.Resize([rnd, rnd],interpolation=Image.NEAREST)(input_tensor)
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        pad_top = randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = randint(0, w_rem)
        pad_right = w_rem - pad_left
        # 要看一下padded的维度来验证  left, top, right and bottom
        padded = transforms.Pad([pad_left, pad_top,pad_right, pad_bottom])(rescaled)

        padded = transforms.Resize([image_width, image_width], interpolation=Image.NEAREST)(padded)

        # padded.set_shape((input_tensor.shape[0], image_resize, image_resize, 3))
        rnd_prob = randint(0,100)/100.0
        if rnd_prob < prob:
            return padded
        else:
            return input_tensor
    else:
        return input_tensor

class ProjectionAttacker(object):

    def __init__(self,
                 args,
                 model,
                 model_prime=None,
                 fc=None,
                 ord='inf',
                 image_width=224,
                 loss_fn=None,
                 targeted=False,
                 rand_init=True):

        self.attack_method = args.attack_method
        self.src_kind = args.src_kind
        self.src_model = args.src_model
        self.use_Inc_model = args.use_Inc_model
        self.model = model
        self.model_prime = model_prime
        self.fc = fc
        self.epsilon = args.epsilon
        self.num_steps = args.num_steps
        self.step_size = args.step_size
        self.linbp_layer = args.linbp_layer
        self.ila_layer = args.ila_layer
        self.ila_niters = args.ila_niters
        self.image_width = image_width
        self.momentum = args.momentum
        self.targeted = targeted
        self.ti_size = args.ti_size
        self.lam = args.lam
        self.grid_scale = args.grid_scale
        self.sample_times = args.sample_times
        if self.ti_size > 1:
            self.ti_conv = transition_invariant_conv(self.ti_size)
        self.sample_grid_num = args.sample_grid_num
        self.m = args.m
        self.sigma = args.sigma
        self.ord = ord
        self.image_resize = args.image_resize
        self.prob = args.prob
        self.rand_init = rand_init
        self.beta = args.beta
        self.number = args.number
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.patch_size = args.patch_size


    def perturb(self, X, y):
        """
        :param X_nat: a Float Tensor  1,c,h,w  float32
        :param y: a Long Tensor 1 int64
        :return:
        """
        loss_record = {'loss1': [], 'loss2': [], 'loss': []}
        delta = torch.zeros_like(X)
        if self.rand_init and self.lam == 0:
            rand_init_delta(delta, X, self.ord, self.epsilon, 0.0, 1.0)
            delta.data = clamp(X + delta.data, min=0.0, max=1.0) - X

        delta.requires_grad_(True)

        grad = torch.zeros_like(X)
        # deltas = torch.zeros_like(X).repeat(self.num_steps, 1, 1, 1)
        label = y.item()
        if self.targeted:
            target = np.random.randint(1000)
            target = (target+label) % 1000
            y[0] = target


        noise_distribution = torch.distributions.normal.Normal(
                    torch.tensor([0.0]),
                    torch.tensor([self.sigma]).float())
        variance = 0.0
        # print(self.model)
        # X_prev = X

        # # DIM attack
        # if self.prob >0:
        #     X = input_diversity(X, self.image_width, self.image_resize, self.prob)

        for i in range(self.num_steps):
            # # DI2 attack
            X_prev = X + delta
            if 'NI' in self.attack_method:
                X_prev = X_prev + self.step_size*self.momentum*grad
            X_DIM = input_diversity(X_prev, self.image_width, self.image_resize, self.prob)

            if 'gfnet' in self.src_kind:
                input_prime = get_prime(X_DIM, self.patch_size)

            if self.m >= 1:  # Variance-reduced attack; https://arxiv.org/abs/1802.09707
                noise_shape = list(X_DIM.shape)
                noise_shape[0] = self.m
                noise = noise_distribution.sample(noise_shape).squeeze() / 255
                noise = noise.to(X_DIM.device)
                outputs = self.model(X_DIM + noise)
                loss1 = self.loss_fn(outputs, y.expand(self.m))
            else:
                if 'gfnet' in self.src_kind:
                    output, state = self.model_prime(input_prime)
                    output = self.fc(output, restart=True)
                    loss1 = self.loss_fn(output, y)
                elif 'vit' in self.src_model:
                    out_adv = self.model(X_DIM)
                    loss1 = 0.0
                    if 'selfensemble' in self.attack_method:
                        for index in range(len(out_adv)):
                            loss1 += self.loss_fn(out_adv[index], y)
                    else:
                        loss1 = self.loss_fn(out_adv[-1], y)

                else:
                    loss1 = self.loss_fn(self.model(X_DIM), y)

            if self.targeted:
                loss1 = -loss1

            if self.lam > 0:  # Interaction-reduced attack
                only_add_one_perturbation, leave_one_out_perturbation = \
                    sample_for_interaction(delta, self.sample_grid_num,
                                           self.grid_scale, self.image_width,
                                           self.sample_times)

                (outputs, leave_one_outputs, only_add_one_outputs,
                 zero_outputs) = get_features(self.model, X, delta,
                                              leave_one_out_perturbation,
                                              only_add_one_perturbation)

                outputs_c = copy.deepcopy(outputs.detach())
                outputs_c[:, label] = -np.inf
                other_max = outputs_c.max(1)[1].item()
                interaction_loss = InteractionLoss(
                    target=other_max, label=label)
                average_pairwise_interaction = interaction_loss(
                    outputs, leave_one_outputs, only_add_one_outputs,
                    zero_outputs)

                if self.lam == float('inf'):
                    loss2 = -average_pairwise_interaction
                    loss = loss2
                else:
                    loss2 = -self.lam * average_pairwise_interaction
                    loss = loss1 + loss2

                loss_record['loss1'].append(loss1.item())
                loss_record['loss2'].append(
                    loss2.item() if self.lam > 0 else 0)
                loss_record['loss'].append(loss.item())
            else:
                loss = loss1
            loss.backward()

            # deltas[i, :, :, :] = delta.data

            cur_grad = delta.grad.data

            if 'VI' in self.attack_method:
                new_grad = cur_grad
                cur_grad = cur_grad + variance
                boundary = self.beta * self.epsilon
                # print(type(boundary))

                global_grad = 0.0
                for num in range(self.number):
                    x_neighbour = X + delta + torch.rand_like(X, dtype=torch.float32).uniform_(-boundary, boundary)
                    x_neighbour = input_diversity(x_neighbour, self.image_width, self.image_resize, self.prob)
                    out = self.model(x_neighbour)
                    if isinstance(out, list):
                        out = out[-1]
                    loss_VI = self.loss_fn(out, y)
                    loss_VI.backward()
                    global_grad += delta.grad.data
                variance = global_grad/(1.0*self.number) - new_grad



            # cur_grad = delta.grad.data
            if self.ti_size > 1:  # TI Attack; https://arxiv.org/abs/1904.02884
                self.ti_conv.to(X.device)
                cur_grad = self.ti_conv(cur_grad)

            # MI Attack; https://arxiv.org/abs/1710.06081
            cur_grad = normalize_by_pnorm(cur_grad, p=1)
            grad = self.momentum * grad + cur_grad

            if self.ord == np.inf:
                delta.data += self.step_size * grad.sign()
                delta.data = clamp(delta.data, -self.epsilon, self.epsilon)
                delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
            elif self.ord == 2:
                delta.data += self.step_size * normalize_by_pnorm(grad, p=2)
                delta.data *= clamp(
                    (self.epsilon * normalize_by_pnorm(delta.data, p=2) /
                     delta.data),
                    max=1.)
                delta.data = clamp(X.data + delta.data, 0.0, 1.0) - X.data
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

            delta.grad.data.zero_()
        # rval = X.data + deltas
        rval = X.data + delta
        if "ila" in self.attack_method:
            ila_model_name = self.src_model
            ila_model_class = model_configs[ila_model_name][1]
            ila_model = load_ila_models(ila_model_name,ila_model_class,self.use_Inc_model)
            source_layers = get_source_layers(self.src_model, ila_model)
            rval = ILA(ila_model, X, X_attack=rval, y=label, feature_layer=source_layers[self.ila_layer][1][1],
                       use_Inc_model=self.use_Inc_model)
            # for layer_ind, (layer_name, layer) in get_source_layers(self.src_model, ila_model):
            #     rval = ILA(ila_model, X, X_attack=rval, y=label, feature_layer=layer,
            #                              use_Inc_model=self.use_Inc_model)
        return rval, loss_record, y.item()

    def perturb_linbp_ila(self, X, y):
        """
        param X_nat: a Float Tensor  1,c,h,w  float32
        :param y: a Long Tensor 1 int64
        :return:
        """
        #
        # model = self.model
        # model.eval()
        # model = nn.Sequential(
        #     Normalize(),
        #     model
        # )
        # model.to(device)
        loss_record = {'loss1': [], 'loss2': [], 'loss': []}

        delta = torch.zeros_like(X)
        if self.rand_init and self.lam == 0:
            rand_init_delta(delta, X, self.ord, self.epsilon, 0.0, 1.0)
            delta.data = clamp(X + delta.data, min=0.0, max=1.0) - X

        X_adv = X + delta
        X_adv.requires_grad_()

        grad = torch.zeros_like(X)

        label = y.item()
        if self.targeted:
            target = np.random.randint(1000)
            target = (target + label) % 1000
            y[0] = target
        # advs = torch.zeros_like(X).repeat(self.num_steps, 1, 1, 1)
        # advs_ila = torch.zeros_like(X).repeat(self.ila_niters, 1, 1, 1)

        noise_distribution = torch.distributions.normal.Normal(
                    torch.tensor([0.0]),
                    torch.tensor([self.sigma]).float())
        # X_prev = X

        # # DIM attack
        # if self.prob >0:
        #     X = input_diversity(X, self.image_width, self.image_resize, self.prob)

        for i in range(self.num_steps):
            # # DI2 attack
            X_DIM = input_diversity(X_adv, self.image_width, self.image_resize, self.prob)
            X_DIM.requires_grad_()
            if self.m >= 1:  # Variance-reduced attack; https://arxiv.org/abs/1802.09707
                noise_shape = list(X_DIM.shape)
                noise_shape[0] = self.m
                noise = noise_distribution.sample(noise_shape).squeeze() / 255
                noise = noise.to(X_DIM.device)
                X_DIM = X_DIM + noise

            if 'linbp' in self.attack_method:
                att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(self.model, X_DIM, True,
                                                                                                    self.linbp_layer)
                pred = torch.argmax(att_out, dim=1).view(-1)
                loss1 = nn.CrossEntropyLoss()(att_out, y)
                if self.targeted:
                    loss1 = -loss1
                self.model.zero_grad()
                cur_grad = linbp_backw_resnet50(X_adv, loss1, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls,
                                                  xp=1.)
            else:
                att_out = self.model(X_DIM)
                # pred = torch.argmax(att_out, dim=1).view(-1)
                loss1 = nn.CrossEntropyLoss()(att_out, y)
                if self.targeted:
                    loss1 = -loss1
                self.model.zero_grad()
                loss1.backward()
                cur_grad = X_adv.grad.data
            self.model.zero_grad()

            # advs[i, :, :, :] = X_adv.data

            # cur_grad = delta.grad.data
            if self.ti_size > 1:  # TI Attack; https://arxiv.org/abs/1904.02884
                self.ti_conv.to(X.device)
                cur_grad = self.ti_conv(cur_grad)

            # MI Attack; https://arxiv.org/abs/1710.06081
            cur_grad = normalize_by_pnorm(cur_grad, p=1)
            grad = self.momentum * grad + cur_grad

            if self.ord == np.inf:
                X_adv.data += self.step_size * grad.sign()
                X_adv.data = clamp(X_adv.data, X-self.epsilon, X+self.epsilon)
                X_adv.data = clamp(X_adv, 0.0, 1.0)
            elif self.ord == 2:
                X_adv.data += self.step_size * normalize_by_pnorm(grad, p=2)
                X_adv.data *= clamp(
                    (self.epsilon * normalize_by_pnorm(X_adv.data, p=2) /
                     X_adv.data),
                    max=1.)
                X_adv.data = clamp(X_adv.data, 0.0, 1.0)
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

        rval = X_adv.data

        # if 'ila' in self.attack_method:
        #     attack_img = X_adv.clone()
        #     X_adv = X.clone().to(device)
        #     with torch.no_grad():
        #         mid_output = ila_forw_resnet50(self.model, X, self.ila_layer)
        #         mid_original = torch.zeros(mid_output.size()).to(device)
        #         mid_original.copy_(mid_output)
        #         mid_output = ila_forw_resnet50(self.model, attack_img, self.ila_layer)
        #         mid_attack_original = torch.zeros(mid_output.size()).to(device)
        #         mid_attack_original.copy_(mid_output)
        #     for _ in range(self.ila_niters):
        #         X_adv.requires_grad_(True)
        #         mid_output = ila_forw_resnet50(self.model, X_adv, self.ila_layer)
        #         loss = ILAProjLoss()(
        #             mid_attack_original.detach(), mid_output, mid_original.detach(), 1.0
        #         )
        #         self.model.zero_grad()
        #         loss.backward()
        #         grad = X_adv.grad.data
        #         self.model.zero_grad()
        #
        #         advs_ila[i, :, :, :] = X_adv.data
        #
        #         # # cur_grad = delta.grad.data
        #         # if self.ti_size > 1:  # TI Attack; https://arxiv.org/abs/1904.02884
        #         #     self.ti_conv.to(X.device)
        #         #     cur_grad = self.ti_conv(cur_grad)
        #         #
        #         # # MI Attack; https://arxiv.org/abs/1710.06081
        #         # cur_grad = normalize_by_pnorm(cur_grad, p=1)
        #         # grad = self.momentum * grad + cur_grad
        #         if self.ord == np.inf:
        #             X_adv.data += self.step_size * grad.sign()
        #             X_adv.data = clamp(X_adv.data, X - self.epsilon, X + self.epsilon)
        #             X_adv.data = clamp(X_adv, 0.0, 1.0)
        #         elif self.ord == 2:
        #             X_adv.data += self.step_size * normalize_by_pnorm(grad, p=2)
        #             X_adv.data *= clamp(
        #                 (self.epsilon * normalize_by_pnorm(X_adv.data, p=2) /
        #                  X_adv.data),
        #                 max=1.)
        #             X_adv.data = clamp(X_adv.data, 0.0, 1.0)
        #         else:
        #             error = "Only ord = inf and ord = 2 have been implemented"
        #             raise NotImplementedError(error)
        #
        #     rval = advs_ila
        #
        #     del mid_output, mid_original, mid_attack_original
        #
        #     # X_adv.grad.data.zero_()
        # rval = advs
        return rval, loss_record, y.item()