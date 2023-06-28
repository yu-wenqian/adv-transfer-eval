
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from codes.model.gfnet.utils import *
from codes.model.gfnet.network import *
from codes.model.gfnet.configs import *

import math
import argparse

import codes.model.gfnet.models.resnet as resnet
import codes.model.gfnet.models.densenet as densenet
from codes.model.gfnet.models import create_model

from codes.dataset import load_gfnet_images

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import csv

# parser = argparse.ArgumentParser(description='Inference code for GFNet')
#
# parser.add_argument('--data_url', default='./data/adv_images/source_resnet50/L_inf_eps_16/PGD+linbp_lam_0.0_seed_0/epoch_99/', type=str,
#                     help='path to the dataset (ImageNet)')
#
# parser.add_argument('--checkpoint_path', default='./checkpoints/resnet50_patch_size_96_T_5.pth.tar', type=str,
#                     help='path to the pre-train model (default: none)')
#
# parser.add_argument('--eval_mode', default=1, type=int,
#                     help='mode 0 : read the evaluation results saved in pre-trained models\
#                           mode 1 : read the confidence thresholds saved in pre-trained models and infer the model on the validation set\
#                           mode 2 : determine confidence thresholds on the training set and infer the model on the validation set')
#
# args = parser.parse_args()


def eval_gfnet(args):
    # load pretrained model
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))

    try:
        model_arch = checkpoint['model_name']
        patch_size = checkpoint['patch_size']
        prime_size = checkpoint['patch_size']
        flops = checkpoint['flops']
        model_flops = checkpoint['model_flops']
        policy_flops = checkpoint['policy_flops']
        fc_flops = checkpoint['fc_flops']
        anytime_classification = checkpoint['anytime_classification']
        budgeted_batch_classification = checkpoint['budgeted_batch_classification']
        dynamic_threshold = checkpoint['dynamic_threshold']
        maximum_length = len(checkpoint['flops'])
    except:
        print('Error: \n'
              'Please provide essential information'
              'for customized models (as we have done '
              'in pre-trained models)!\n'
              'At least the following information should be Given: \n'
              '--model_name: name of the backbone CNNs (e.g., resnet50, densenet121)\n'
              '--patch_size: size of image patches (i.e., H\' or W\' in the paper)\n'
              '--flops: a list containing the Multiply-Adds corresponding to each '
              'length of the input sequence during inference')

    model_configuration = model_configurations[model_arch]

    if args.eval_mode > 0:
        # create model
        if 'resnet' in model_arch:
            model = resnet.resnet50(pretrained=False)
            model_prime = resnet.resnet50(pretrained=False)            

        elif 'densenet' in model_arch:
            model = eval('densenet.' + model_arch)(pretrained=False)
            model_prime = eval('densenet.' + model_arch)(pretrained=False)

        elif 'efficientnet' in model_arch:
            model = create_model(model_arch, pretrained=False, num_classes=1000,
                                    drop_rate=0.3, drop_connect_rate=0.2)
            model_prime = create_model(model_arch, pretrained=False, num_classes=1000,
                                    drop_rate=0.3, drop_connect_rate=0.2)
        
        elif 'mobilenetv3' in model_arch:
            model = create_model(model_arch, pretrained=False, num_classes=1000,
                                    drop_rate=0.2, drop_connect_rate=0.2)
            model_prime = create_model(model_arch, pretrained=False, num_classes=1000,
                                    drop_rate=0.2, drop_connect_rate=0.2)
        
        elif 'regnet' in model_arch:
            import codes.model.gfnet.pycls.core.model_builder as model_builder
            from codes.model.gfnet.pycls.core.config import cfg
            cfg.merge_from_file(model_configuration['cfg_file'])
            cfg.freeze()

            model = model_builder.build_model()
            model_prime = model_builder.build_model()

        # traindir = args.data_url + 'train/'
        valdir = args.data_url

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        # train_set = datasets.ImageFolder(traindir, transforms.Compose([
        #         transforms.RandomResizedCrop(model_configuration['image_size'], interpolation=model_configuration['dataset_interpolation']),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize ]))
        # train_set_index = torch.randperm(len(train_set))
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, num_workers=32, pin_memory=False,
        #         sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-200000:]))

        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(valdir, transforms.Compose([
        #         transforms.Resize(int(model_configuration['image_size']/model_configuration['crop_pct']), interpolation = model_configuration['dataset_interpolation']),
        #         transforms.CenterCrop(model_configuration['image_size']),
        #         transforms.ToTensor(),
        #         normalize])),
        #     batch_size=256, shuffle=False, num_workers=16, pin_memory=False)
        val_loader, _ = load_gfnet_images.load_images(
            input_dir=valdir,
            image_size=model_configuration['image_size'],
            crop_pct=model_configuration['crop_pct'],
            dataset_interpolation=model_configuration['dataset_interpolation'],
            batch_size=100)

        state_dim = model_configuration['feature_map_channels'] * math.ceil(patch_size/32) * math.ceil(patch_size/32)
        
        memory = Memory()
        policy = ActorCritic(model_configuration['feature_map_channels'], state_dim, model_configuration['policy_hidden_dim'], model_configuration['policy_conv'])
        fc = Full_layer(model_configuration['feature_num'], model_configuration['fc_hidden_dim'], model_configuration['fc_rnn'])

        model = nn.DataParallel(model.to(device))
        model_prime = nn.DataParallel(model_prime.to(device))
        policy = policy.to(device)
        fc = fc.to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        policy.load_state_dict(checkpoint['policy'])

        budgeted_batch_flops_list = []
        budgeted_batch_acc_list = []

        print('generate logits on test samples...')
        test_logits, test_targets, anytime_classification = generate_logits(model_prime, model, fc, memory, policy, val_loader, maximum_length, prime_size, patch_size, model_arch)
        
        # if args.eval_mode == 2:
        #     print('generate logits on training samples...')
        #     dynamic_threshold = torch.zeros([39, maximum_length])
        #     train_logits, train_targets, _ = generate_logits(model_prime, model, fc, memory, policy, train_loader, maximum_length, prime_size, patch_size, model_arch)

        for p in range(1, 2):

            print('inference: {}/40'.format(p))

            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, maximum_length))
            probs /= probs.sum()

            # if args.eval_mode == 2:
            #     dynamic_threshold[p-1] = dynamic_find_threshold(train_logits, train_targets, probs)
            
            acc_step, flops_step, logits = dynamic_evaluate(test_logits, test_targets, flops, dynamic_threshold[p-1])
            
            budgeted_batch_acc_list.append(acc_step)
            budgeted_batch_flops_list.append(flops_step)
        
        budgeted_batch_classification = [budgeted_batch_flops_list, budgeted_batch_acc_list]

    print('model_arch :', model_arch)
    print('patch_size :', patch_size)
    print('flops :', flops)
    print('model_flops :', model_flops)
    print('policy_flops :', policy_flops)
    print('fc_flops :', fc_flops)
    print('anytime_classification :', anytime_classification)
    print('budgeted_batch_classification :', budgeted_batch_classification)

def eval_gfnet_ensemble(args):
    # load pretrained model
    checkpoint = torch.load(args.checkpoint_path)
    anytime_csv_path = 'gfnet_experiments/gfnet1.csv'
    budget_csv_path = 'gfnet_experiments/gfnet2.csv'

    try:
        model_arch = checkpoint['model_name']
        patch_size = checkpoint['patch_size']
        prime_size = checkpoint['patch_size']
        flops = checkpoint['flops']
        model_flops = checkpoint['model_flops']
        policy_flops = checkpoint['policy_flops']
        fc_flops = checkpoint['fc_flops']
        anytime_classification = checkpoint['anytime_classification']
        budgeted_batch_classification = checkpoint['budgeted_batch_classification']
        dynamic_threshold = checkpoint['dynamic_threshold']
        maximum_length = len(checkpoint['flops'])
    except:
        print('Error: \n'
              'Please provide essential information'
              'for customized models (as we have done '
              'in pre-trained models)!\n'
              'At least the following information should be Given: \n'
              '--model_name: name of the backbone CNNs (e.g., resnet50, densenet121)\n'
              '--patch_size: size of image patches (i.e., H\' or W\' in the paper)\n'
              '--flops: a list containing the Multiply-Adds corresponding to each '
              'length of the input sequence during inference')

    model_configuration = model_configurations[model_arch]

    if args.eval_mode > 0:
        # create model
        if 'resnet' in model_arch:
            model = resnet.resnet50(pretrained=False)
            model_prime = resnet.resnet50(pretrained=False)

        elif 'densenet' in model_arch:
            model = eval('densenet.' + model_arch)(pretrained=False)
            model_prime = eval('densenet.' + model_arch)(pretrained=False)

        elif 'efficientnet' in model_arch:
            model = create_model(model_arch, pretrained=False, num_classes=1000,
                                 drop_rate=0.3, drop_connect_rate=0.2)
            model_prime = create_model(model_arch, pretrained=False, num_classes=1000,
                                       drop_rate=0.3, drop_connect_rate=0.2)

        elif 'mobilenetv3' in model_arch:
            model = create_model(model_arch, pretrained=False, num_classes=1000,
                                 drop_rate=0.2, drop_connect_rate=0.2)
            model_prime = create_model(model_arch, pretrained=False, num_classes=1000,
                                       drop_rate=0.2, drop_connect_rate=0.2)

        elif 'regnet' in model_arch:
            import codes.model.gfnet.pycls.core.model_builder as model_builder
            from codes.model.gfnet.pycls.core.config import cfg
            cfg.merge_from_file(model_configuration['cfg_file'])
            cfg.freeze()

            model = model_builder.build_model()
            model_prime = model_builder.build_model()

        # traindir = args.data_url + 'train/'
        valdir = args.data_url

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        # train_set = datasets.ImageFolder(traindir, transforms.Compose([
        #         transforms.RandomResizedCrop(model_configuration['image_size'], interpolation=model_configuration['dataset_interpolation']),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize ]))
        # train_set_index = torch.randperm(len(train_set))
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, num_workers=32, pin_memory=False,
        #         sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-200000:]))

        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(valdir, transforms.Compose([
        #         transforms.Resize(int(model_configuration['image_size']/model_configuration['crop_pct']), interpolation = model_configuration['dataset_interpolation']),
        #         transforms.CenterCrop(model_configuration['image_size']),
        #         transforms.ToTensor(),
        #         normalize])),
        #     batch_size=256, shuffle=False, num_workers=16, pin_memory=False)
        val_loader, _ = load_gfnet_images.load_images(
            input_dir=valdir,
            image_size=model_configuration['image_size'],
            crop_pct=model_configuration['crop_pct'],
            dataset_interpolation=model_configuration['dataset_interpolation'],
            batch_size=100)

        state_dim = model_configuration['feature_map_channels'] * math.ceil(patch_size / 32) * math.ceil(
            patch_size / 32)

        memory = Memory()
        policy = ActorCritic(model_configuration['feature_map_channels'], state_dim,
                             model_configuration['policy_hidden_dim'], model_configuration['policy_conv'])
        fc = Full_layer(model_configuration['feature_num'], model_configuration['fc_hidden_dim'],
                        model_configuration['fc_rnn'])

        model = nn.DataParallel(model.to(device))
        model_prime = nn.DataParallel(model_prime.to(device))
        policy = policy.to(device)
        fc = fc.to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        policy.load_state_dict(checkpoint['policy'])

        budgeted_batch_flops_list = []
        budgeted_batch_acc_list = []
        budgeted_logits_list = []

        print('generate logits on test samples...')
        target_class = np.load(args.target_root + "/target_class.npy", allow_pickle=True).item()
        test_logits, test_targets, anytime_classification = generate_logits(model_prime, model, fc, memory, policy,
                                                                            val_loader, maximum_length, prime_size,
                                                                            patch_size, model_arch, target_class)
        with open(anytime_csv_path, 'a+', encoding='utf-8') as f:
            # 调用模块的类：模块名.类名（）
            csv_writer = csv.writer(f)
            # 调用类中的方法：对象.方法名（）
            csv_writer.writerow(flops)
            # csv_writer.writerow(model_flops)
            # csv_writer.writerow(fc_flops)
            # csv_writer.writerow(policy_flops)
            csv_writer.writerow(anytime_classification)
        for p in range(1, 40):
            # print('inference: {}/40'.format(p))

            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, maximum_length))
            probs /= probs.sum()

            # if args.eval_mode == 2:
            #     dynamic_threshold[p-1] = dynamic_find_threshold(train_logits, train_targets, probs)

            acc_step, flops_step, logits = dynamic_evaluate(test_logits, test_targets, flops, dynamic_threshold[p - 1])

            budgeted_batch_acc_list.append(acc_step)
            budgeted_batch_flops_list.append(flops_step)
            budgeted_logits_list.append(logits)

        with open(budget_csv_path, 'a+', encoding='utf-8') as f:
            # 调用模块的类：模块名.类名（）
            csv_writer = csv.writer(f)
            # 调用类中的方法：对象.方法名（）
            csv_writer.writerow(budgeted_batch_flops_list)
            csv_writer.writerow(budgeted_batch_acc_list)

        budgeted_batch_classification = [budgeted_batch_flops_list, budgeted_batch_acc_list]

    print('model_arch :', model_arch)
    print('patch_size :', patch_size)
    print('flops :', flops)
    print('model_flops :', model_flops)
    print('policy_flops :', policy_flops)
    print('fc_flops :', fc_flops)
    print('anytime_classification :', anytime_classification)
    print('budgeted_batch_classification :', budgeted_batch_classification)

    return budgeted_logits_list[0], test_targets


def generate_logits(model_prime, model, fc, memory, policy, dataloader, maximum_length, prime_size, patch_size, model_arch, target_class):

    logits_list = []
    targets_list = []

    top1 = [AverageMeter() for _ in range(maximum_length)]
    model.eval()
    model_prime.eval()
    fc.eval()

    target_labels = torch.randint(1, [100])  # 第二个值是batchsize

    for (x, target, file_names) in dataloader:

        # print(file_names)

        logits_temp = torch.zeros(maximum_length, x.size(0), 1000)
        target_labels = torch.randint(1, [100])

        target_index = 0
        for filename in file_names:
            target_labels[target_index] = target_class[filename]
            target_index = target_index + 1

        target_var = target.to(device)
        target_labels = target_labels.to(device)
        input_var = x.to(device)

        input_prime = get_prime(input_var, prime_size, model_configurations[model_arch]['prime_interpolation'])

        with torch.no_grad():

            output, state = model_prime(input_prime)

            if 'resnet' in model_arch or 'densenet' in model_arch:
                output = fc(output, restart=True)
            elif 'regnet' in model_arch:
                _ = fc(output, restart=True)
                output = model_prime.module.fc(output)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            logits_temp[0] = F.softmax(output, 1)
            acc = accuracy(output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))

            for patch_step in range(1, maximum_length):

                with torch.no_grad():
                    if patch_step == 1:
                        action = policy.act(state, memory, restart_batch=True)
                    else:
                        action = policy.act(state, memory)

                patches = get_patch(input_var, action, patch_size)
                output, state = model(patches)
                output = fc(output, restart=False)

                logits_temp[patch_step] = F.softmax(output, 1)
                acc = accuracy(output, target_var, topk=(1,))
                top1[patch_step].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))

        logits_list.append(logits_temp)
        # targets_list.append(target_var)
        targets_list.append(target_labels)

        memory.clear_memory()

        anytime_classification = []

        for index in range(maximum_length):
            anytime_classification.append(top1[index].ave)

    return torch.cat(logits_list, 1), torch.cat(targets_list, 0), anytime_classification


def dynamic_find_threshold(logits, targets, p):

    n_stage, n_sample, c = logits.size()
    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)
    _, sorted_idx = max_preds.sort(dim=1, descending=True)

    filtered = torch.zeros(n_sample)
    T = torch.Tensor(n_stage).fill_(1e8)

    for k in range(n_stage - 1):
        acc, count = 0.0, 0
        out_n = math.floor(n_sample * p[k])
        for i in range(n_sample):
            ori_idx = sorted_idx[k][i]
            if filtered[ori_idx] == 0:
                count += 1
                if count == out_n:
                    T[k] = max_preds[k][ori_idx]
                    break
        filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

    T[n_stage - 1] = -1e8
    return T
   

def dynamic_evaluate(logits, targets, flops, T):

    n_stage, n_sample, c = logits.size()
    logits_temp = torch.zeros(n_sample, c)
    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)
    _, sorted_idx = max_preds.sort(dim=1, descending=True)

    acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
    acc, expected_flops = 0, 0
    for i in range(n_sample):
        gold_label = targets[i]
        for k in range(n_stage):
            if max_preds[k][i].item() >= T[k]:  # force the sample to exit at k
                if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                    acc += 1
                    acc_rec[k] += 1
                exp[k] += 1
                logits_temp[i] = logits[k][i]
                break
    acc_all = 0
    for k in range(n_stage):
        _t = 1.0 * exp[k] / n_sample
        expected_flops += _t * flops[k]
        acc_all += acc_rec[k]

    return acc * 100.0 / n_sample, expected_flops.item(), logits_temp


# if __name__ == '__main__':
#     main()
