# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/28  下午10:34
# File Name: train.py
# IDE: PyCharm
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import random
from math import cos, pi
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import torch.distributed as dist

from config.config import get_parser
import tools.log as log


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# #val_set visual
def vis_pred(render_img, lm_68n_pre, lm_68n, front_raw, fn_1):
    # # plt
    render_image_plt = render_img.detach().cpu().numpy()
    keypoint_face_pre_plt = lm_68n_pre.detach().cpu().numpy()
    keypoint_face_pts_plt = lm_68n.detach().cpu().numpy()
    render_image_plt = render_image_plt[0, :, :, :]
    front_raw_filter_plt = front_raw.detach().cpu().numpy()
    front_raw_filter_plt = front_raw_filter_plt[0, :, :, :]
    plt.subplot(1, 2, 1)
    render_image_plt[render_image_plt < 0] = 0
    render_image_plt[render_image_plt > 1] = 1
    plt.imshow(render_image_plt)
    plt.scatter(keypoint_face_pre_plt[0, :, 0], keypoint_face_pre_plt[0, :, 1], c='r', s=2)
    plt.scatter(keypoint_face_pts_plt[0, :, 0], keypoint_face_pts_plt[0, :, 1], c='b', s=2)
    plt.subplot(1, 2, 2)
    render_image_plt[front_raw_filter_plt < 0] = 0
    render_image_plt[front_raw_filter_plt > 1] = 1
    plt.imshow(front_raw_filter_plt)
    plt.scatter(keypoint_face_pts_plt[0, :, 0], keypoint_face_pts_plt[0, :, 1], c='b', s=2)
    # plt.show()
    # #
    os.makedirs('./result/val/', exist_ok=True)  # create dir
    plt.savefig('./result/val/' + fn_1 + '.png')
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.subplot(1, 2, 2)
    plt.cla()
    pass


def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    model.train()

    # #for log the run time and remain time
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()  # initialization
    am_dict = {}

    # #start train
    for i, batch in enumerate(train_loader):
        # torch.cuda.empty_cache()
        batch_time.update(time.time() - end_time)  # update time

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs)  # adjust lr

        # #loss, result, visual_dict , meter_dict (visual_dict: tensorboardX, meter_dict: average batch loss)
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        # # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #average batch loss, time for print
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = log.AverageMeter()
            am_dict[k].update(v[0], v[1])

        current_iter = epoch * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            sys.stdout.write("epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) "
                             "iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                             .format(epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val,
                                     am_dict['loss'].avg,
                                     batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
                                     remain_time=remain_time))
            if (i == len(train_loader) - 1): print()

    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                         time.time() - start_time))
        # #write tensorboardX
        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_train', am_dict[k].avg, epoch)

        # # save pretrained model
        pretrain_file = log.checkpoint_save(model, cfg.logpath, epoch, cfg.save_freq)
        logger.info('Saving {}'.format(pretrain_file))

    pass


def eval_epoch(val_loader, model, model_fn, epoch):
    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        for i, batch in enumerate(val_loader):
            # torch.cuda.empty_cache()
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)
            # # visual
            if epoch % 10 == 0:
                render_img_left = preds['render_img_lt']
                render_img_front = preds['render_img_ft']
                render_img_right = preds['render_img_rt']
                kp_img_pred_lt = preds['kp_img_pred_lt']
                kp_img_pred_ft = preds['kp_img_pred_ft']
                kp_img_pred_rt = preds['kp_img_pred_rt']
                raw_left = preds['raw_lt']
                raw_front = preds['raw_ft']
                raw_right = preds['raw_rt']
                kp_img_gt_lt = preds['kp_img_gt_lt']
                kp_img_gt_ft = preds['kp_img_gt_ft']
                kp_img_gt_rt = preds['kp_img_gt_rt']
                file_name = preds['file_name']
                fn_0 = file_name[0][:-1] + '0'
                fn_1 = file_name[0][:-1] + '1'
                fn_2 = file_name[0][:-1] + '2'
                if epoch!=0: os.removedirs('./result/val/')
                vis_pred(render_img_left, kp_img_pred_lt, kp_img_gt_lt, raw_left, fn_0)
                vis_pred(render_img_front, kp_img_pred_ft, kp_img_gt_ft, raw_front, fn_1)
                vis_pred(render_img_right, kp_img_pred_rt, kp_img_gt_rt, raw_right, fn_2)

            # #average batch loss, time for print
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = log.AverageMeter()
                am_dict[k].update(v[0], v[1])
            if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
                sys.stdout.write(
                    "\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val,
                                                                am_dict['loss'].avg))
                if (i == len(val_loader) - 1): print()
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                           time.time() - start_time))
            # #write tensorboardX
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


def Distributed_training(gpu, cfgs):
    global cfg
    cfg = cfgs
    cfg.local_rank = gpu
    # logger and summary write
    if cfg.local_rank == 0:
        # logger
        global logger
        from tools.log import get_logger
        logger = get_logger(cfg)
        logger.info(cfg)  # log config
        # summary writer
        global writer
        writer = SummaryWriter(cfg.logpath)
    cfg.rank = cfg.node_rank * cfg.gpu_per_node + gpu
    print('[PID {}] rank: {}  world_size: {}'.format(os.getpid(), cfg.rank, cfg.world_size))
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % cfg.tcp_port, world_size=cfg.world_size,
                            rank=cfg.rank)
    if cfg.local_rank == 0:
        logger.info(cfg)
    # #set cuda
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    if cfg.local_rank == 0:
        logger.info('cuda available: {}'.format(use_cuda))

    # #create model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')
    from network.MVRnet import MVRNet as net
    from network.MVRnet import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg)
    model = model.to(gpu)
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    if cfg.local_rank == 0:
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)
    # load dataset
    if cfg.dataset == 'pixel_face':
        from dataset.pixel_face.dataset_preprocess import Dataset
    else:
        print('Only support pixel_face dataset now')

    dataset = Dataset(cfg)
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info('Training samples: {}'.format(len(dataset.train_file_list)))
        logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, cfg.logpath, dist=cfg.dist, pretrain_file=cfg.pretrain,
                                                        gpu=gpu)
    if cfg.local_rank == 0:
        logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                    else 'Start from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, cfg.epochs):
        dataset.train_sampler.set_epoch(epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        # #validation
        if cfg.validation:
            dataset.val_sampler.set_epoch(epoch)
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
    pass


def Single_card_training(gpu, cfg):
    # #logger
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # #summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    # # create model
    logger.info('=> creating model ...')
    from network.MVRnet import MVRNet as net
    from network.MVRnet import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    model = net(cfg)
    model = model.to(gpu)
    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)

    # load dataset
    if cfg.dataset == 'pixel_face':
        from dataset.pixel_face.dataset_preprocess import Dataset
    else:
        print('Only support pixel_face dataset now')

    dataset = Dataset(cfg)
    dataset.trainLoader()
    dataset.valLoader()
    logger.info('Training samples: {}'.format(len(dataset.train_file_list)))
    logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, cfg.logpath, pretrain_file=cfg.pretrain, gpu=gpu)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, cfg.epochs):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)
        #
        # # #validation
        if cfg.validation:
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)

    pass


if __name__ == '__main__':

    # # fix seed for debug
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)  # cpu
    torch.cuda.manual_seed(22)

    # #INIT
    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
    cfg = get_parser()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.manual_seed)

    # # Determine whether it is distributed training
    cfg.world_size = cfg.nodes * cfg.gpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    if cfg.dist:
        mp.spawn(Distributed_training, nprocs=cfg.gpu_per_node, args=(cfg,))
    else:
        Single_card_training(cfg.local_rank, cfg)
