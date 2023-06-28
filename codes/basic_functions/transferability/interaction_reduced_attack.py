import os

import numpy as np
import torch
import torch.nn as nn
from codes.basic_functions.transferability.get_attacker import get_attacker
from codes.dataset.load_images import load_images
from codes.dataset.save_images import save_images
from codes.model.load_model import load_imagenet_model, load_vit_model, load_snn_model, load_gfnet_model
from codes.model.normalizer import Normalize
from codes.utils import reset_dir
from tqdm import tqdm

from codes.model.gfnet.inference import eval_gfnet_ensemble
from codes.model.gfnet.utils import get_prime


def generate_adv_images(args):
    print(f'Source arch {args.src_model}')
    print(f'Source kind {args.src_kind}')
    print(f'Attack: {args.attack_method}')
    # print(f'Attack kind: {args.tar_kind}')
    print(
        f'Args: momentum {args.momentum}, gamma {args.gamma}, m {args.m}, sigma {args.sigma}, grid {args.sample_grid_num}, times {args.sample_times}, lam {args.lam}'
    )

    if 'cnn' in args.src_kind:
        model = load_imagenet_model(model_type=args.src_model)
        mean, std = model.mean, model.std
        height, width = model.input_size[1], model.input_size[2]
        predict = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)
    elif 'vit' in args.src_kind:
        model, mean, std = load_vit_model(args)
        height = args.image_size
        width = args.image_size
        predict = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)
    elif 'snn' in args.src_kind:
        height = args.image_size
        width = args.image_size
        adv_dataloader, _ = load_images(
            input_dir=args.clean_image_root,
            input_height=height,
            input_width=width)
        predict = load_snn_model(args, adv_dataloader).to(args.device)
    elif 'gfnet' in args.src_kind:
        height = args.image_size
        width = args.image_size
        adv_dataloader, _ = load_images(
            input_dir=args.clean_image_root,
            input_height=height,
            input_width=width)
        predict, model_prime, fc = load_gfnet_model(args)
        predict = predict.to(args.device)
        model_prime = model_prime.to(args.device)
        fc = fc.to(args.device)

    # height, width = model.input_size[1], model.input_size[2]
    # mean, std = model.mean, model.std
    # predict = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)

    ori_dataloader, _ = load_images(
        input_dir=args.clean_image_root,
        input_height=height,
        input_width=width)

    if 'gfnet' in args.src_kind:
        adversary = get_attacker(args, predict=predict, model_prime=model_prime, fc=fc, image_dim=height * width * 3,
                                 image_size=height)
    else:
        adversary = get_attacker(args, predict=predict, model_prime=None, fc=None, image_dim=height * width * 3,
                                 image_size=height)

    save_root = args.adv_image_root
    reset_dir(save_root)
    # for epoch in range(args.num_steps):
    #     epoch_save_root = os.path.join(save_root, f'epoch_{epoch}')
    #     reset_dir(epoch_save_root)
    epoch_save_root = os.path.join(save_root, f'epoch_{args.num_steps - 1}')
    reset_dir(epoch_save_root)
    reset_dir(args.loss_root)
    reset_dir(args.target_root)
    loss_record = {}
    target_class = {}

    for (ori_image, label, file_names) in tqdm(ori_dataloader):

        image_index = 0

        ori_image = ori_image.to(args.device)
        with torch.no_grad():
            if 'gfnet' in args.src_kind:
                input_prime = get_prime(ori_image, args.patch_size)
                out, state = model_prime(input_prime)
                out = fc(out, restart=True)
                pred = out.max(1)[1].item()
            else:
                out = predict(ori_image)
                if isinstance(out, list):
                    out = out[-1]
                pred = out.max(1)[1].item()
            if label.item() != pred:
                print('Invalid prediction')
                # raise Exception('Invalid prediction')

        if 'linbp' in args.attack_method:
            advs, loss_record_i, target = adversary.perturb_linbp_ila(
                ori_image,
                torch.tensor([label]).to(args.device),
            )
        else:
            advs, loss_record_i, target = adversary.perturb(
                ori_image,
                torch.tensor([label]).to(args.device),
            )

        # advs_len = len(advs)
        advs_len = args.num_steps - 1

        advs = advs.detach().cpu().numpy()
        file_name = file_names[0]
        # for epoch in range(advs_len):
        #     epoch_save_root = os.path.join(save_root, f'epoch_{epoch}')
        #     save_images(
        #         images=advs[epoch:epoch + 1, :, :, :],
        #         filename=file_name,
        #         output_dir=epoch_save_root)
        epoch_save_root = os.path.join(save_root, f'epoch_{advs_len}')
        save_images(
            images=advs,
            filename=file_name,
            output_dir=epoch_save_root)
        loss_record[file_name] = loss_record_i
        target_class[file_name] = target

        image_index = image_index + 1

    np.save(os.path.join(args.loss_root, 'loss_record.npy'), loss_record)
    np.save(os.path.join(args.target_root, 'target_class.npy'), target_class)

def eval_ensemble(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    # vit_list = ["vit_base_patch16_224", "vit_base_patch8_224", "vit_base_patch32_224",
    #             "vit_small_patch32_224", "vit_small_patch16_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]
    model_num = 18.0
    batch = 100
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    # adv_dir = args.clean_image_root
    args.data_url = adv_dir
    logits_sum = torch.zeros([5000, 1000]).to(args.device)
    logits_temp = torch.zeros([5000, 1000]).to(args.device)
    logits_path = os.path.join('./experiments/logits', setting, f'epoch_{epoch}')
    if not os.path.exists(logits_path):
        os.makedirs(logits_path)
    softmax = nn.Softmax(dim=1)

    target_class = np.load(args.target_root + "/target_class.npy", allow_pickle=True).item()
    target_labels = torch.randint(1, [batch]).to(args.device)  # 第二个值是batchsize

    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        model = load_imagenet_model(model_type=cnn)
        mean, std = model.mean, model.std
        height, width = model.input_size[1], model.input_size[2]
        print(f'{cnn} success loaded')
        model = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)
        adv_dataloader, _ = load_images(
            input_dir=adv_dir,
            input_height=height,
            input_width=width,
            batch_size=batch)
        i = 0
        acc = 0.0
        for (adv_images, labels, file_names) in adv_dataloader:
            adv_images = adv_images.to(args.device)

            target_index = 0
            for filename in file_names:
                target_labels[target_index] = target_class[filename]
                target_index = target_index + 1

            with torch.no_grad():
                # acc = 0.0
                logits = softmax(model(adv_images)).to(args.device)
                logits_temp[i*batch: (i+1)*batch] = logits
                logits_sum[i*batch: (i+1)*batch] = logits + logits_sum[i*batch: (i+1)*batch]
                i = i+1
                for j in range(batch):
                    pred_j = logits[j:j + 1].max(1)[1].item()
                    label_j = target_labels[j].item()
                    if pred_j == label_j:
                        acc += 1.0
        target_sucess = acc/5000.0
        print(f'{cnn} target success rate: {target_sucess}')
        torch.save(logits_temp, logits_temp_path)
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        args.tar_model = vit
        model, mean, std = load_vit_model(args)
        height = args.image_size
        width = args.image_size
        print(f'{vit} success loaded')
        model = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)
        adv_dataloader, _ = load_images(
            input_dir=adv_dir,
            input_height=height,
            input_width=width,
            batch_size=batch)
        i = 0
        acc = 0.0
        for (adv_images, labels, file_names) in adv_dataloader:
            adv_images = adv_images.to(args.device)

            target_index = 0
            for filename in file_names:
                target_labels[target_index] = target_class[filename]
                target_index = target_index + 1

            with torch.no_grad():
                # acc = 0.0
                out = model(adv_images)
                if isinstance(out, list):
                    out = out[-1]
                logits = softmax(out).to(args.device)
                logits_temp[i * batch: (i + 1) * batch] = logits
                logits_sum[i * batch: (i + 1) * batch] = logits + logits_sum[i * batch: (i + 1) * batch]
                i = i + 1
                for j in range(batch):
                    pred_j = logits[j:j + 1].max(1)[1].item()
                    label_j = target_labels[j].item()
                    if pred_j == label_j:
                        acc += 1.0
        target_sucess = acc / 5000.0
        print(f'{vit} target success rate: {target_sucess}')
        torch.save(logits_temp, logits_temp_path)
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        args.tar_model = snn
        height = args.image_size
        width = args.image_size
        adv_dataloader, _ = load_images(
            input_dir=adv_dir,
            input_height=height,
            input_width=width,
            batch_size=batch)
        model = load_snn_model(args, adv_dataloader).to(args.device)
        print(f'{snn}{args.calib} success loaded')
        i = 0
        acc = 0.0
        for (adv_images, labels, file_names) in adv_dataloader:
            adv_images = adv_images.to(args.device)

            target_index = 0
            for filename in file_names:
                target_labels[target_index] = target_class[filename]
                target_index = target_index + 1

            with torch.no_grad():
                logits = softmax(model(adv_images)).to(args.device)
                logits_temp[i * batch: (i + 1) * batch] = logits
                logits_sum[i * batch: (i + 1) * batch] = logits + logits_sum[i * batch: (i + 1) * batch]
                i = i + 1
                for j in range(batch):
                    pred_j = logits[j:j + 1].max(1)[1].item()
                    label_j = target_labels[j].item()
                    if pred_j == label_j:
                        acc += 1.0
        target_sucess = acc/5000.0
        print(f'{snn}{args.calib} target success rate: {target_sucess}')
        torch.save(logits_temp, logits_temp_path)
    del model
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        args.tar_model = snn
        height = args.image_size
        width = args.image_size
        adv_dataloader, _ = load_images(
            input_dir=adv_dir,
            input_height=height,
            input_width=width,
            batch_size=batch)
        model = load_snn_model(args, adv_dataloader).to(args.device)
        print(f'{snn}{args.calib} success loaded')
        i = 0
        acc = 0.0
        for (adv_images, labels, file_names) in adv_dataloader:
            adv_images = adv_images.to(args.device)

            target_index = 0
            for filename in file_names:
                target_labels[target_index] = target_class[filename]
                target_index = target_index + 1

            with torch.no_grad():
                logits = softmax(model(adv_images)).to(args.device)
                logits_temp[i * batch: (i + 1) * batch] = logits
                logits_sum[i * batch: (i + 1) * batch] = logits + logits_sum[i * batch: (i + 1) * batch]
                i = i + 1
                for j in range(batch):
                    pred_j = logits[j:j + 1].max(1)[1].item()
                    label_j = target_labels[j].item()
                    if pred_j == label_j:
                        acc += 1.0
        target_sucess = acc/5000.0
        print(f'{snn}{args.calib} target success rate: {target_sucess}')
        torch.save(logits_temp, logits_temp_path)
    del model
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        args.checkpoint_path = checkpoint
        logits, target_labels = eval_gfnet_ensemble(args)
        logits  = logits.to(args.device)
        acc = 0.0
        for j in range(5000):
            pred_j = logits[j:j + 1].max(1)[1].item()
            label_j = target_labels[j].item()
            if pred_j == label_j:
                acc += 1.0
        target_sucess = acc / 5000.0
        print(f'{checkpoint} target success rate: {target_sucess}')
        logits_sum = logits + logits_sum
        torch.save(logits, logits_temp_path)
        gfnet_index = gfnet_index + 1


def ensemble_result(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]

    model_num = 18.0
    batch = 1
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    args.data_url = adv_dir
    logits_sum = torch.zeros([5000, 1000])
    logits_path = os.path.join('./experiments/logits', setting, f'epoch_{epoch}')
    # if not os.path.exists(logits_path):
    #     os.makedirs(logits_path)
    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        logits_temp = torch.load(logits_temp_path, map_location=torch.device('cpu'))
        logits_sum = logits_temp + logits_sum
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        logits_temp = torch.load(logits_temp_path, map_location=torch.device('cpu'))
        logits_sum = logits_temp + logits_sum
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path, map_location=torch.device('cpu'))
        logits_sum = logits_temp + logits_sum
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path, map_location=torch.device('cpu'))
        logits_sum = logits_temp + logits_sum
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        logits_temp = torch.load(logits_temp_path, map_location=torch.device('cpu'))
        logits_sum = logits_temp + logits_sum
        gfnet_index = gfnet_index + 1

    logits_sum = logits_sum / model_num

    target_class = np.load(args.target_root + "/target_class.npy", allow_pickle=True).item()
    target_labels = torch.randint(1, [batch])  # 第二个值是batchsize

    adv_dataloader, _ = load_images(
        input_dir=adv_dir,
        input_height=224,
        input_width=224,
        batch_size=batch)
    i = 0
    acc = 0.0
    for (adv_image, label, file_names) in adv_dataloader:

        target_index = 0
        for filename in file_names:
            target_labels[target_index] = target_class[filename]
            target_index = target_index + 1

        with torch.no_grad():
            for j in range(batch):
                pred_j = logits_sum[batch * i + j: batch * i + j + 1].max(1)[1].item()
                label_j = target_labels[j].item()
                if pred_j == label_j:
                    acc += 1.0
        i = i + 1
    sucess_rate = acc / i
    print(f'ensemble1 target success rate: {sucess_rate}')


def test_protocol3(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]
    model_num = 18
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    args.data_url = adv_dir
    logits_sum = torch.zeros([model_num, 5000, 1000])
    logits_path = os.path.join('./experiments/logits', setting, f'epoch_{epoch}')
    # if not os.path.exists(logits_path):
    #     os.makedirs(logits_path)
    i = 0
    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        logits_temp = torch.load(logits_temp_path,)
        logits_sum[i] = logits_temp
        i = i + 1
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
        gfnet_index = gfnet_index + 1

    adv_dataloader, _ = load_images(
        input_dir=adv_dir,
        input_height=224,
        input_width=224,
        batch_size=1)
    i = 0
    acc = 0.0

    for (adv_image, label, file_name) in adv_dataloader:
        with torch.no_grad():
            label_i = label.item()
            for j in range(model_num):
                pred_i = logits_sum[j][i:i + 1].max(1)[1].item()
                if pred_i == label_i:
                    break
            else:
                acc += 1.0
        i = i + 1
    sucess_rate = acc / i
    print(f'ensemble3 success rate: {sucess_rate}')


def target_test_protocol3(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]
    model_num = 18
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    args.data_url = adv_dir
    logits_sum = torch.zeros([model_num, 5000, 1000])
    logits_path = os.path.join('./target_experiments/logits', setting, f'epoch_{epoch}')
    # if not os.path.exists(logits_path):
    #     os.makedirs(logits_path)
    i = 0
    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
        gfnet_index = gfnet_index + 1

    adv_dataloader, _ = load_images(
        input_dir=adv_dir,
        input_height=224,
        input_width=224,
        batch_size=1)
    i = 0
    acc = 0.0

    target_class = np.load(args.target_root + "/target_class.npy", allow_pickle=True).item()

    for (adv_image, label, file_name) in adv_dataloader:
        with torch.no_grad():
            label_i = target_class[file_name[0]]
            for j in range(model_num):
                pred_i = logits_sum[j][i:i + 1].max(1)[1].item()
                if pred_i != label_i:
                    break
            else:
                acc += 1.0
        i = i + 1
    sucess_rate = acc / i
    print(f'ensemble3 target success rate: {sucess_rate}')


def test_protocol2(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]
    model_num = 18
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    args.data_url = adv_dir
    logits_sum = torch.zeros([model_num, 5000, 1000])
    logits_path = os.path.join('./experiments/logits', setting, f'epoch_{epoch}')
    # if not os.path.exists(logits_path):
    #     os.makedirs(logits_path)
    i = 0
    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
        gfnet_index = gfnet_index + 1

    adv_dataloader, _ = load_images(
        input_dir=adv_dir,
        input_height=224,
        input_width=224,
        batch_size=1)
    i = 0
    acc = 0.0

    for (adv_image, label, file_name) in adv_dataloader:
        with torch.no_grad():
            label_i = label.item()
            for j in range(model_num):
                pred_i = logits_sum[j][i:i + 1].max(1)[1].item()
                if pred_i != label_i:
                    acc = acc + 1.0
        i = i + 1
    success_rate = acc / i
    print(f'avg_transfer_models: {success_rate}')


def target_test_protocol2(args):
    setting = os.path.join(
        f'source_{args.src_model}', f'L_{args.p}_eps_{args.epsilon}',
        '_'.join([args.attack_method, f'lam_{args.lam}_seed_{args.seed}']))
    cnn_list = ["vgg16", "resnet50", "densenet121", "inceptionv3"]
    vit_list = ["vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224",
                "deit_base_patch16_224", "deit_tiny_patch16_224", "deit_small_patch16_224",
                "swin_base_patch4_window7_224"]
    snn_list = ["vgg16", "res34"]
    gfnet_checkpoint_list = ["./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/resnet50_patch_size_128_T_5.pth.tar",
                             "./codes/model/gfnet/checkpoints/densenet121_patch_size_96_T_5.pth.tar"]
    model_num = 18
    epoch = args.num_steps - 1
    adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')
    args.data_url = adv_dir
    logits_sum = torch.zeros([model_num, 5000, 1000])
    logits_path = os.path.join('./target_experiments/logits', setting, f'epoch_{epoch}')
    # if not os.path.exists(logits_path):
    #     os.makedirs(logits_path)
    i = 0
    for cnn in cnn_list:
        logits_temp_path = os.path.join(logits_path, f'{cnn}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for vit in vit_list:
        logits_temp_path = os.path.join(logits_path, f'{vit}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    for snn in snn_list:
        args.calib = 'light'
        logits_temp_path = os.path.join(logits_path, f'snn_{snn}_{args.calib}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
    gfnet_index = 1
    for checkpoint in gfnet_checkpoint_list:
        logits_temp_path = os.path.join(logits_path, f'gfnet_{gfnet_index}.pt')
        logits_temp = torch.load(logits_temp_path)
        logits_sum[i] = logits_temp
        i = i + 1
        gfnet_index = gfnet_index + 1

    adv_dataloader, _ = load_images(
        input_dir=adv_dir,
        input_height=224,
        input_width=224,
        batch_size=1)
    i = 0
    acc = 0.0

    target_class = np.load(args.target_root + "/target_class.npy", allow_pickle=True).item()

    for (adv_image, label, file_name) in adv_dataloader:
        with torch.no_grad():
            label_i = target_class[file_name[0]]
            for j in range(model_num):
                pred_i = logits_sum[j][i:i + 1].max(1)[1].item()
                if pred_i == label_i:
                    acc += 1.0
        i = i + 1
    success_rate = acc / i
    print(f'avg_transfer_models: {success_rate}')


