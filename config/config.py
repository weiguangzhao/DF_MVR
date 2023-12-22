# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/7  上午10:10
# File Name: config.py
# IDE: PyCharm

import argparse

# config para
def get_parser():
    parser = argparse.ArgumentParser(description='3D Face Reconstruction')
    # # Basic Setting
    parser.add_argument('--task', type=str, default='train', help='task: train or test')
    parser.add_argument('--manual_seed', type=int, default=22, help='seed to produce')
    parser.add_argument('--cache', type=bool, default=True, help='Whether to use shm')
    parser.add_argument('--epochs', type=int, default=280, help='Total epoch')
    parser.add_argument('--first_time', type=bool, default=True, help='first time to generate npy data')
    parser.add_argument('--mask_pred_path', type=str, default='pretrain/face_mask.pth',
                        help='path to pretrain model of face mask')
    parser.add_argument('--logpath', type=str, default='log/z_101_3090/', help='path to save logs')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--mask_epoch', type=int, default=100, help='Mask epoch')

    # # Dataset setting
    parser.add_argument('--dataset', type=str, default='pixel_face', help='choose dataset: pixel_face, MICC')
    parser.add_argument('--batch_size_train', type=int, default=12, help='batch_size_train for single GPU')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch_size_val for single GPU')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset')
    parser.add_argument('--num_train_sample', type=int, default=None,
                        help='select the number of train samples, none is all')
    parser.add_argument('--save_freq', type=int, default=8, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--validation', type=bool, default=True, help='Whether to verify the validation set')

    # #Adjust learning rate
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--step_epoch', type=int, default=50, help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.5, help='Learning rate decay: lr = lr * multiplier')

    # #Distributed training
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('-nr', '--node_rank', type=int, default=0, help='ranking within the nodes')
    parser.add_argument('--nodes', type=int, default=1, help='Number of distributed training nodes')
    parser.add_argument('--gpu_per_node', type=int, default=2, help='Number of GPUs per node')
    parser.add_argument('--sync_bn', type=bool, default=False, help='Whether to batch norm all gpu para')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for Distributed training')

    args = parser.parse_args()
    return args
