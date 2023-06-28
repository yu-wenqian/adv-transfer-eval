import torch
import pretrainedmodels
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from codes.basic_functions.ouradvertorch.utils import clamp

model_configs = {
    "resnet18": ("resnet18", models.resnet18),
    "densenet121": ("densenet121", models.densenet121),
    "squeezenet1_0": ("squeezenet1_0", models.squeezenet1_0),
    "alexnet": ("alexnet", models.alexnet),
    'inceptionv3': ('inceptionv3',pretrainedmodels.__dict__['inceptionv3']),
    'inceptionresnetv2': ('inceptionresnetv2',pretrainedmodels.__dict__['inceptionresnetv2']),
    'inceptionv4': ('inceptionv4',pretrainedmodels.__dict__['inceptionv4']),
}

def get_source_layers(model_name, model):
    if model_name == 'resnet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)),
                                  ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'])))

    elif model_name == 'densenet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)),
                              ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3',
                               'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))

    elif model_name == 'squeezenet1_0':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer ' + name, model._modules.get('features')._modules.get(name)),
                              ['0', '3', '4', '5', '7', '8', '9', '10', '12']))
        layer_list.append(('classifier', model._modules.get('classifier')._modules.get('1')))
        return list(enumerate(layer_list))

    elif model_name == 'alexnet':
        # exclude avgpool
        layer_list = list(map(lambda name: ('layer ' + name, model._modules.get('features')._modules.get(name)),
                              ['0', '3', '6', '8', '10']))
        layer_list += list(
            map(lambda name: ('layer ' + name, model._modules.get('classifier')._modules.get(name)), ['1', '4', '6']))
        return list(enumerate(layer_list))

    elif model_name == 'inceptionresnetv2':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)),
                                  ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a',
                                   'maxpool_5a', 'mixed_5b', 'repeat', 'mixed_6a', 'repeat_1', 'mixed_7a', 'repeat_2',
                                   'block8', 'conv2d_7b', 'avgpool_1a', 'last_linear'])))

    elif model_name == 'inceptionv4':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)),
                              ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                               '16', '17', '18', '19', '20', '21']))
        return list(enumerate(layer_list))

    elif model_name == 'inceptionv3':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get(name)),
                              ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                               'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c']))
        return list(enumerate(layer_list))

    else:
        # model is not supported
        assert False

# ILA attack

# square sum of dot product
class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


# square sum of dot product
class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff


"""Return: perturbed x"""
mid_output = None


def normalize_and_scale_imagenet(delta_im, epsilon, use_Inc_model):
    """Normalize and scale imagenet perturbation according to epsilon Linf norm
    Args:
        delta_im: perturbation on imagenet images
        epsilon: Linf norm
    Returns:
        The re-normalized perturbation
    """

    if use_Inc_model:
        stddev_arr = [0.5, 0.5, 0.5]
    else:
        stddev_arr = [0.229, 0.224, 0.225]

    for ci in range(3):
        mag_in_scaled = epsilon / stddev_arr[ci]
        delta_im[:, ci] = delta_im[:, ci].clone().clamp(-mag_in_scaled, mag_in_scaled)

    return delta_im

def renormalization(X, X_pert, epsilon, dataset="cifar10", use_Inc_model = False):
    """Normalize and scale perturbations according to epsilon Linf norm
    Args:
        X: original images
        X_pert: adversarial examples corresponding to X
        epsilon: Linf norm
        dataset: dataset images are from, 'cifar10' | 'imagenet'
    Returns:
        The re-normalized perturbation
    """
    # make sure you don't modify the original image beyond epsilon, also clamp
    if dataset == "cifar10":
        eps_added = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        # clamp
        return eps_added.clamp(-1.0, 1.0)
    elif dataset == "imagenet":
        eps_added = normalize_and_scale_imagenet(X_pert.detach() - X.clone(), epsilon, use_Inc_model) + X.clone()
        # clamp
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for i in range(3):
            min_clamp = (0 - mean[i]) / stddev[i]
            max_clamp = (1 - mean[i]) / stddev[i]
            eps_added[:,i] = eps_added[:,i].clone().clamp(min_clamp, max_clamp)
        return eps_added

def ILA(
    model,
    X,
    X_attack,
    y,
    feature_layer,
    niters=10,
    epsilon=16.0/255.0,
    coeff=1.0,
    learning_rate=2.0/255.0,
    dataset="imagenet",
    use_Inc_model = False,
    with_projection=True,
):
    """Perform ILA attack with respect to model on images X with labels y
    Args:
        with_projection: boolean, specifies whether projection should happen
        in the attack
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        X_attack: starting adversarial examples of ILA that will be modified
        to become more transferable
        y: labels corresponding to the batch of images
        feature_layer: layer of model to project on in ILA attack
        niters: number of iterations of the attack to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        coeff: coefficient of magnitude loss in ILA attack
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of the attack
        dataset: dataset the images are from, 'cifar10' | 'imagenet'
    Returns:
        The batch of modified adversarial examples, examples have been
        augmented from X_attack to become more transferable
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).to(device)
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        # print(m)
        # print(i)
        # print(o)
        global mid_output
        mid_output = o
        # print(mid_output)

    h = feature_layer.register_forward_hook(get_mid_output)

    out = model(X)
    mid_original = torch.zeros(mid_output.size()).to(device)
    mid_original.copy_(mid_output)

    out = model(X_attack)
    mid_attack_original = torch.zeros(mid_output.size()).to(device)
    mid_attack_original.copy_(mid_output)

    for _ in range(niters):
        output_perturbed = model(X_pert)

        # generate adversarial example by max middle layer pertubation
        # in the direction of increasing loss
        if with_projection:
            loss = Proj_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )
        else:
            loss = Mid_layer_target_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )

        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # minimize loss
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon
        # X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        # X_pert.requires_grad = True
        X_pert.data = clamp(X_pert.data, X.data-epsilon, X.data+epsilon)
        X_pert.data = clamp(X_pert.data, 0.0, 1.0)
        X_pert.requires_grad = True

    h.remove()
    return X_pert

