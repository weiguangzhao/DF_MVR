# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/7  上午10:48
# File Name: log.py
# IDE: PyCharm

import os
import time
import logging
import glob
import torch


# Used to calculate elapsed time and estimated time
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)  # filename: build a FileHandler
    return logger


def get_logger(cfg):
    if cfg.task == 'train':
        log_file = os.path.join(
            cfg.logpath, 'train',
            'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'test':
        log_file = os.path.join(
            cfg.logpath, 'result', 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file)
    logger.info('************************ Start Logging ************************')

    return logger


def checkpoint_restore(model, logpath, epoch=0, dist=False, pretrain_file='', gpu=0):
    if not pretrain_file:
        if epoch > 0:
            pretrain_file = os.path.join(logpath + '%09d' % epoch + '.pth')
            assert os.path.isfile(pretrain_file)
        else:
            pretrain_file = sorted(glob.glob(os.path.join(logpath + '*.pth')))
            if len(pretrain_file) > 0:
                pretrain_file = pretrain_file[-1]
                epoch = int(pretrain_file[len(logpath) + 2: -4])

    if len(pretrain_file) > 0:
        map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
        checkpoint = torch.load(pretrain_file, map_location=map_location)
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        epoch = int(pretrain_file[len(logpath) + 2: -4])

    return epoch, pretrain_file


def checkpoint_save(model, logpath, epoch, save_freq=1):
    pretrain_file = os.path.join(logpath + '%09d' % epoch + '.pth')
    torch.save(model.state_dict(), pretrain_file)
    # remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    fd = os.path.join(logpath + '%09d' % epoch + '.pth')
    if os.path.isfile(fd):
        if epoch % save_freq != 0:
            os.remove(fd)
    return pretrain_file


