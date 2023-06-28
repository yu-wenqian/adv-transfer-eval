import argparse

from codes.basic_functions.transferability import (interaction_reduced_attack)
from set_config import set_config

from codes.model.gfnet.inference import eval_gfnet

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p", type=str, help="inf; 2", default='inf')
parser.add_argument("--epsilon", type=int, default=16)
parser.add_argument("--step_size", type=float, default=2.)
parser.add_argument("--num_steps", type=int, default=40)
parser.add_argument("--loss_root", type=str, default='./experiments/loss')
parser.add_argument("--target_root", type=str, default='./experiments/target')
parser.add_argument(
    "--adv_image_root", type=str, default='./experiments/adv_images')
parser.add_argument("--clean_image_root", type=str, default='data/images_5000')
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--src_model", type=str, default='vit_small_patch8_224')
parser.add_argument("--src_kind", type=str, default='vit')
parser.add_argument(
    "--tar_model", type=str, default='vit_small_patch8_224')
parser.add_argument("--tar_kind", type=str, default='vit')

# args for different attack methods
parser.add_argument("--attack_method", type=str, default='PGD40')
# args for generate or evaluate mode
parser.add_argument('--generate_mode', action='store_true', help='generate adv examples or just evaluate')

parser.add_argument("--gamma", type=float, default=1.)
parser.add_argument("--momentum", type=float, default=0.)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--sigma", type=float, default=15.)
parser.add_argument("--ti_size", type=int, default=1)
parser.add_argument("--lam", type=float, default=0.)
parser.add_argument("--grid_scale", type=int, default=16)
parser.add_argument("--sample_grid_num", type=int, default=32)
parser.add_argument("--sample_times", type=int, default=32)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_resize", type=int, default=255)
parser.add_argument("--prob", type=float, default=0.)
parser.add_argument('--linbp_layer', type=str, default='3_1')
parser.add_argument('--ila_layer', type=int, default=6)
parser.add_argument('--ila_niters', type=int, default=10)
#args for ila
parser.add_argument('--use_Inc_model', action='store_true', help='<Required> use Inception models group')
#args for VI
parser.add_argument("--beta", type=float, default=1.5)
parser.add_argument('--number', type=int, default=20)

# args for vit models
# ghost vit/deit
parser.add_argument('--skip_eros', default=0.01, type=float)
parser.add_argument('--drop_layer', default=0.01, type=float)
# tokenvit
parser.add_argument('--token_combi', default='', type=str)

#args for snn models
parser.add_argument('--T', default=16, type=int, help='snn simulation length')
parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])

# args for gfnet
parser.add_argument('--data_url', default='/experiments/adv_images/source_inceptionv3/L_inf_eps_16/PGD40_lam_0.0_seed_0/epoch_39/', type=str,
                    help='path to the dataset (ImageNet)')

parser.add_argument("--patch_size", type=int, default=96)

parser.add_argument('--checkpoint_path', default='./codes/model/gfnet/checkpoints/resnet50_patch_size_96_T_5.pth.tar',
                    type=str,
                    help='path to the pre-train model (default: none)')

parser.add_argument('--eval_mode', default=1, type=int,
                    help='mode 0 : read the evaluation results saved in pre-trained models\
                          mode 1 : read the confidence thresholds saved in pre-trained models and infer the model on the validation set\
                          mode 2 : determine confidence thresholds on the training set and infer the model on the validation set')

parser.add_argument('--device', default='cuda:0', type=str)
# parser.add_argument('--device', default='cpu', type=str)

args = parser.parse_args()

def ttest_interaction_reduced_attack():
    set_config(args)
    if args.generate_mode:
        interaction_reduced_attack.generate_adv_images(args)
    else:
        if "gfnet" in args.tar_kind:
            eval_gfnet(args)
        elif "ensemble" in args.tar_model:
            interaction_reduced_attack.eval_ensemble(args) #individual models target FR:output; untarget FR:1.0-output
            interaction_reduced_attack.ensemble_result(args) #protocol1 target FR:output; untarget FR:1.0-output
            interaction_reduced_attack.test_protocol2(args) #protocol2 untarget
            # interaction_reduced_attack.target_test_protocol2(args) #protocol2 target
            interaction_reduced_attack.test_protocol3(args) #protocol3 untarget
            # interaction_reduced_attack.target_test_protocol3(args) #protocol3 target


ttest_interaction_reduced_attack()
