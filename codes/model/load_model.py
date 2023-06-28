import pretrainedmodels
import torch.nn as nn
import torchvision.models as models
import torch
from codes.model.imagenet_ensemble import ImagenetEnsemble
from codes.model.utils import util_vit

from codes.model.snn_models.calibration import bias_corr_model, weights_cali_model
from codes.model.snn_models.fold_bn import search_fold_and_remove_bn
from codes.model.snn_models.ImageNet.models.mobilenet import mobilenetv1
from codes.model.snn_models.ImageNet.models.resnet import res_spcials, resnet34_snn
from codes.model.snn_models.ImageNet.models.vgg import vgg16, vgg16_bn, vgg_specials
from codes.model.snn_models.spiking_layer import SpikeModel,get_maximum_activation
from codes.model.snn_models.distributed_utils import get_local_rank, initialize

import codes.model.gfnet.models.resnet as resnet
import codes.model.gfnet.models.densenet as densenet
from codes.model.gfnet.network import *
from codes.model.gfnet.configs import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_imagenet_model(model_type):
    if model_type == 'ensemble':
        model = ImagenetEnsemble()
    else:
        model = pretrainedmodels.__dict__[model_type](
            num_classes=1000, pretrained='imagenet').eval()
        for param in model.parameters():
            param.requires_grad = False
    model.eval()
    return model

def load_vit_model(args):
    (tar_model, _), tar_mean, tar_std = util_vit.get_model(args.tar_model, args)
    tar_model = tar_model.to(device)
    tar_model.eval()
    return tar_model,tar_mean,tar_std

def load_ila_models(source_model_name,model_class,use_Inc_model):
    # model_name = args.src_model
    # for source_model_name, model_class in ila_models:
    #     if use_Inc_model:
    #         model = model_class[source_model_name](num_classes=1000, pretrained='imagenet').to(device)
    #     else:
    #         model = model_class(pretrained=True).to(device)
    if use_Inc_model:
        model = model_class(num_classes=1000, pretrained='imagenet').to(device)
    else:
        model = model_class(pretrained=True).to(device)
    model.eval()
    return model

def load_snn_model(args,dataloader):
    try:
        initialize()
        initialized = True
        torch.cuda.set_device(get_local_rank())
    except:
        print('For some reason, your distributed environment is not initialized, this program may run on separate GPUs')
        initialized = False

    sim_length = 32

    if args.tar_model == 'vgg16':
        ann = vgg16_bn(pretrained=True) if args.usebn else vgg16(pretrained=True)
    elif args.tar_model == 'res34':
        ann = resnet34_snn(pretrained=True, use_bn=args.usebn)
    elif args.tar_model == 'mobilenet':
        ann = mobilenetv1(pretrained=True)
    else:
        raise NotImplementedError

    search_fold_and_remove_bn(ann)
    ann.to(device)

    snn = SpikeModel(model=ann, sim_length=sim_length,
                     specials=vgg_specials if args.tar_model == 'vgg16' else res_spcials)
    snn.to(device)

    mse = False if args.calib == 'none' else True

    # initialized = True
    # import torch.distributed as dist
    # dist.init_process_group('gloo', rank=0, world_size=1)
    get_maximum_activation(dataloader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None,
                           sim_length=sim_length, channel_wise=False, dist_avg=initialized)

    # make sure dist_avg=True to synchronize the data in different GPUs, e.g. gradient and threshold
    # otherwise each gpu performs its own calibration

    if args.calib == 'light':
        bias_corr_model(model=snn, train_loader=dataloader, correct_mempot=False, dist_avg=initialized)
    if args.calib == 'advanced':
        weights_cali_model(model=snn, train_loader=dataloader, batch_size=100, num_cali_samples=1000,
                           learning_rate=1e-5, dist_avg=initialized)
        bias_corr_model(model=snn, train_loader=dataloader,
                        correct_mempot=True, dist_avg=initialized)

    snn.set_spike_state(use_spike=True)
    snn.eval()
    # print(snn)
    return snn

def load_gfnet_model(args):
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model_arch = checkpoint['model_name']
    model_configuration = model_configurations[model_arch]
    if 'resnet' in model_arch:
        model = resnet.resnet50(pretrained=False)
        model_prime = resnet.resnet50(pretrained=False)
    elif 'densenet' in model_arch:
        model = eval('densenet.' + model_arch)(pretrained=False)
        model_prime = eval('densenet.' + model_arch)(pretrained=False)
    fc = Full_layer(model_configuration['feature_num'], model_configuration['fc_hidden_dim'],
                    model_configuration['fc_rnn'])
    model = nn.DataParallel(model.to(device))
    model_prime = nn.DataParallel(model_prime.to(device))
    fc = fc.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
    fc.load_state_dict(checkpoint['fc'])
    model.eval()
    model_prime.eval()
    fc.eval()

    return model,model_prime,fc
