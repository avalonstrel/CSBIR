import torch
from torch.backends import cudnn
import os
import argparse
from models import networks
from datasets import datasets
from utils import makedir, save_config
from train import Solver
import json

def str2bool(s):
    return s.lower() == 'true'

# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_root', type=str, required=1)
    parser.add_argument('--obj', type=str, default='TUBerlin')
    parser.add_argument('--crop_size', type=int, default=225)
    parser.add_argument('--seed', type=int, default=666)

    # phase
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--flag', type=str, default='sbir')
    parser.add_argument('--mode', type=str, default='std')

    # model
    parser.add_argument('--model_type', type=str, default='densenet169')
    parser.add_argument('--distance', type=str, default='sq')
    parser.add_argument('--feat_dim', type=int, default=512)

    # train
    parser.add_argument('--pretrain_num_steps', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--num_steps_decay', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--pretrained_model', type=int, default=-1)

    parser.add_argument('--loss_type', type=str, default='triplet')
    parser.add_argument('--loss_ratio', type=str, default='1.0')

    # log
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--valid_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--print_log', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=False)

    parser.add_argument('--root', type=str, default='expr')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--model_path', type=str, default='model')

    config = parser.parse_args()

    # device
    #config.device = torch.device('cuda:0')
    config.loss_type = config.loss_type.split(',')
    config.loss_ratio = [eval(r) for r in config.loss_ratio.split(',')]    

    # log
    try:
        config.log_path = os.path.join(config.root, config.log_path)
    except FileExistsError:
        pass
    config.model_path = os.path.join(config.root, config.model_path)
    makedir(config.log_path)
    makedir(config.model_path)

    return config

def main():
    config = get_parameter()
    save_config(config)
    cudnn.benchmark = True
    config.device = torch.device('cuda:0')
    ##### build model #####
    model_type = config.model_type.split('_')
    norm = False
    if len(model_type) == 2:
        norm = str2bool(model_type[1])
    model = {'net':networks[model_type[0]](f_dim=config.feat_dim, norm=norm)}

    ##### create dataset #####
    seed = config.seed if config.seed > 0 else None
    data = datasets[config.obj](config.data_root, config.crop_size, seed=seed)
    config.c_dim = data.num_cates

    ##### train/test #####
    solver = Solver(config, data, model)
    solver.train()


if __name__ == '__main__':
    main()
