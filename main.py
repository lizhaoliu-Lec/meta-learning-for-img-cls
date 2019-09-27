import argparse
import sys
from pprint import pprint

import torch

from utils import set_gpu
from trainers import PreTrainer


def mtl_argparser():
    parser = argparse.ArgumentParser(prog='mtl_argparser')

    basic_group = parser.add_argument_group('basic')
    pretrain_group = parser.add_argument_group('pretrain')
    meta_group = parser.add_argument_group('meta')

    # basic settings
    basic_group.add_argument('--basenet_name', type=str, default='mini_ss_resnet50',
                             choices=['mini_ss_resnet12',
                                      'mini_ss_resnet18',
                                      'mini_ss_resnet50',
                                      'mini_ss_resnet34',
                                      'mini_ss_resnet101', ],
                             help='select the basenet architecture.')
    basic_group.add_argument('--dataset_basedir', type=str, default='./data',
                             help='where the base dir for all datasets.')
    basic_group.add_argument('--dataset_name', type=str, default='mini',
                             choices=['mini', 'FC100', 'tiered'],
                             help='dataset name to train the model.')
    basic_group.add_argument('--phase', type=str, default='pretrain',
                             choices=['pretrain', 'meta_train', 'preval', 'meta_eval'],
                             help='phase to use the model.')
    basic_group.add_argument('--gpu', type=str, default='0',
                             help='select the gpu to train the model, -1'
                                  ' means train on cpu.')
    basic_group.add_argument('--seed', type=int, default=0,
                             help='manual seed for pytorch.')
    basic_group.add_argument('--save_dir', type=str, default='./checkpoints',
                             help='base dir for saving the model.')
    basic_group.add_argument('--mode', type=str, default='pre',
                             choices=['pre', 'meta'],
                             help='select the mode to train or eval the model.')
    basic_group.add_argument('--log_dir', type=str, default='./experiments',
                             help='path to save the log.')

    pretrain_tail = '_pre'
    # for pretrain the model
    pretrain_group.add_argument('--max_epoch' + pretrain_tail, type=int, default=100,
                                help='max epoch number for pretrain the model.')
    pretrain_group.add_argument('--batch_size' + pretrain_tail, type=int, default=128,
                                help='batch size for pretrain the model.')
    pretrain_group.add_argument('--num_batch' + pretrain_tail, type=int, default=100,
                                help='number of different tasks for meta eval the model.')
    pretrain_group.add_argument('--shot' + pretrain_tail, type=int, default=1,
                                help='how many number of samples for one class in a task when eval the model.')
    pretrain_group.add_argument('--way' + pretrain_tail, type=int, default=5,
                                help='how many classes in a task when eval a model.')
    pretrain_group.add_argument('--query' + pretrain_tail, type=int, default=15,
                                help='how many number of samples to query for one class in a task.')
    pretrain_group.add_argument('--lr' + pretrain_tail, type=float, default=0.1,
                                help='learning rate for pretrain the model.')
    pretrain_group.add_argument('--gamma' + pretrain_tail, type=float, default=0.2,
                                help='gamma learning rate decay.')
    pretrain_group.add_argument('--step_size' + pretrain_tail, type=int, default=30,
                                help='number of epoch to decay learning rate.')
    pretrain_group.add_argument('--momentum' + pretrain_tail, type=float, default=0.9,
                                help='momentum fot SGD optimizer.')
    pretrain_group.add_argument('--weight_decay' + pretrain_tail, type=float, default=0.0005,
                                help='weight decay for SGD optimizer.')
    pretrain_group.add_argument('--fast_lr' + pretrain_tail, type=float, default=0.01,
                                help='fast learning rate for fast train the query examples.')
    pretrain_group.add_argument('--fast_step' + pretrain_tail, type=int, default=50,
                                help='fast learning step for fast train the query examples.')
    pretrain_group.add_argument('--resume' + pretrain_tail, action='store_true',
                                help='whether to resume the model to train.')
    pretrain_group.add_argument('--save_dir' + pretrain_tail, type=str, default='pre',
                                help='sub dir for saving the pretrained model.')
    pretrain_group.add_argument('--time' + pretrain_tail, type=str, default='',
                                help='last time to pre train the model.')

    meta_tail = '_meta'
    # for meta train or eval the model
    meta_group.add_argument('--max_epoch' + meta_tail, type=int, default=100,
                            help='max epoch number for meta train the model.')
    meta_group.add_argument('--num_batch' + meta_tail, type=int, default=100,
                            help='number of different tasks for meta train the model.')
    meta_group.add_argument('--shot' + meta_tail, type=int, default=1,
                            help='how many number of samples for one class in a task.')
    meta_group.add_argument('--way' + meta_tail, type=int, default=5,
                            help='how many classes in a task.')
    meta_group.add_argument('--query' + meta_tail, type=int, default=15,
                            help='how many number of samples to query for one class in a task.')
    meta_group.add_argument('--lr' + meta_tail, type=float, default=0.001,
                            help='learning rate for meta train the model in meta mode.')
    meta_group.add_argument('--gamma' + meta_tail, type=float, default=0.2,
                            help='gamma learning rate decay.')
    meta_group.add_argument('--step_size' + meta_tail, type=int, default=30,
                            help='number of epoch to decay learning rate.')
    meta_group.add_argument('--momentum' + meta_tail, type=float, default=0.9,
                            help='momentum fot SGD optimizer.')
    meta_group.add_argument('--weight_decay' + meta_tail, type=float, default=0.0005,
                            help='weight decay for SGD optimizer.')
    meta_group.add_argument('--fast_lr' + meta_tail, type=float, default=0.01,
                            help='fast learning rate for fast train the query examples.')
    meta_group.add_argument('--fast_step' + meta_tail, type=int, default=50,
                            help='fast learning step for fast train the query examples.')
    meta_group.add_argument('--resume' + meta_tail, action='store_true',
                            help='whether to resume the model to train.')
    meta_group.add_argument('--save_dir' + meta_tail, type=str, default='meta',
                            help='sub dir for saving the meta trained model.')
    meta_group.add_argument('--time' + meta_tail, type=str, default='',
                            help='last time to meta train the model.')

    return parser.parse_args()


if __name__ == '__main__':

    # get args
    args = mtl_argparser()
    pprint(vars(args))

    # set gpu
    set_gpu(args.gpu)

    # set seed and deterministic
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # four phases
    # pretrain, meta_train, preval, meta_eval
    if args.phase == 'meta_train':
        # trainer = MetaTrainer(args)
        # trainer.train()
        sys.exit(0)
    if args.phase == 'meta_eval':
        # trainer = MetaTrainer(args)
        # trainer.eval()
        sys.exit(0)
    if args.phase == 'pretrain':
        trainer = PreTrainer(args)
        trainer.train()
        sys.exit(0)
    if args.phase == 'preval':
        # trainer = PreTrainer(args)
        # trainer.eval()
        sys.exit(0)

    raise ValueError('Please select correct phase.')
