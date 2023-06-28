import numpy as np
import torch.nn as nn
from codes.basic_functions.ouradvertorch.attacks.projected_attack import \
    ProjectionAttacker
from codes.utils import register_hook_for_densenet, register_hook_for_resnet
from codes.model.vit_models.utils import register_hook_for_vit

def get_attacker(args,predict, model_prime,fc,image_dim, image_size):

    if ('SGM' in args.attack_method or 'Hybrid' in args.attack_method) and args.gamma > 1:
        raise Exception('gamma of SGM method should be less than 1')
    if ('MI' in args.attack_method or 'Hybrid' in args.attack_method) and args.momentum == 0:
        raise Exception('momentum of MI method should be greater than 0')
    if ('VR' in args.attack_method or 'Hybrid' in args.attack_method) and args.m == 0:
        raise Exception('m of VR method should be greater than 0')
    if ('TI' in args.attack_method) and args.ti_size == 1:
        raise Exception('ti_size of  the TI method should be greater than 0')
    if ('IR' in args.attack_method or 'Hybrid' in args.attack_method) and args.lam == 0:
        raise Exception('lam of  the IR method should be greater than 0')

    if args.p == 'inf':
        args.p = np.inf
        args.epsilon = args.epsilon / 255.
        args.step_size = args.step_size / 255.
        args.num_steps = args.num_steps
    elif int(args.p) == 2:
        args.p = 2
        args.epsilon = args.epsilon / 255. * np.sqrt(image_dim)
        args.step_size = float(args.step_size)
        args.num_steps = args.num_steps
    else:
        raise NotImplementedError('p should be inf or 2')

    # set for SGM Attack
    if args.gamma < 1.0 and ('SGM' in args.attack_method or 'Hybrid' in args.attack_method):
        if args.src_model in [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        ]:
            register_hook_for_resnet(predict, arch=args.src_model, gamma=args.gamma)
        elif args.src_model in ['densenet121', 'densenet169', 'densenet201']:
            register_hook_for_densenet(predict, arch=args.src_model, gamma=args.gamma)
        else:
            raise ValueError(
                'Current code only supports resnet/densenet. '
                'You can extend this code to other architectures.')

    #set for PNA attack --vit_base
    if 'PNA' in args.attack_method:
        register_hook_for_vit(predict[1], arch=args.src_model)


    adversary = ProjectionAttacker(args,
                                   model=predict,
                                   model_prime=model_prime,
                                   fc=fc,
                                   ord=args.p,
                                   image_width=image_size,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                   targeted=False,
                                   )
    return adversary
