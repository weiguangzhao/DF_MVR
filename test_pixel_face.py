# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/28  下午10:34
# File Name: test_pixel_face.py
# IDE: PyCharm
import os
import sys
import time
import torch
import glob
from config.config_test import get_parser
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from tools.render import Renderer
import tools.log as log
from lib.BFM.model_basis.BFM2009.face_generate import Face3D
from tools.plt import save_obj
# from lib.now_evaluation.compute_error import compute_rmse


def save_pred(preds, cfg):
    # # visual
    result_path_obj = cfg.result_path + 'obj/'
    result_path_detail = cfg.result_path + 'detail/'
    os.makedirs(result_path_obj, exist_ok=True)  # create dir
    os.makedirs(result_path_detail, exist_ok=True)  # create dir

    face_shape_ft = preds['face_shape_ft']
    vertices_img = preds['xyz_project_ft']
    face_texture = preds['face_texture']
    file_name = preds['file_name']
    fn = file_name[0][:-1]
    face_model = Face3D()

    # # save 3d obj
    face_obj = face_model.facemodel.cell
    vertices_img_obj = face_shape_ft[0, ...].detach().cpu().numpy()
    # vertices_img_color_obj = face_texture[0, :, :].detach().cpu().numpy()*255.0
    vertices_img_color_obj = np.ones_like(vertices_img_obj)*127.5
    save_obj(vertices_img_obj, vertices_img_color_obj, face_obj+1, result_path_obj + fn + '.obj')
    pass



def eval_epoch(val_loader, model, model_fn, cfg):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}
    rmse_npy = []
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            file_name = batch['file_name']
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch=20)
            # # save 3d obj
            save_pred(preds, cfg)
            print('saved {}'.format(i))

            # #####calulate rmse
            # raw_ft = preds['raw_ft']
            # face_shape_ft = preds['face_shape_ft']
            # face_texture = preds['face_texture']
            # fn = file_name[0][:-1]
            # face_model = Face3D()
            # face_obj = face_model.facemodel.cell
            # face_obj = face_obj.detach().cpu().numpy()
            # vertices_img_obj = face_shape_ft[0, ...].detach().cpu().numpy()
            # rmse = compute_rmse(vertices_img_obj, face_obj, fn)
            # rmse_npy.append(rmse)
            # visual_dict['rmse'] = rmse
            # meter_dict['rmse'] = (rmse.item(), raw_ft.shape[0])

            # #average batch loss, time for print
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = log.AverageMeter()
                am_dict[k].update(v[0], v[1])
            if (i == len(val_loader) - 1): print()


def Single_card_testing(gpu, cfg):
    torch.cuda.set_device(gpu)
    # #logger
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # #summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    # #create model
    logger.info('=> creating model ...')
    from network.MVRnet import MVRNet as net
    from network.MVRnet import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg)
    model = model.to(gpu)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # load dataset
    if cfg.dataset == 'pixel_face':
        from dataset.pixel_face.dataset_preprocess import Dataset
    else:
        print('Only support pixel_face dataset now')

    dataset = Dataset(cfg)
    dataset.valLoader()
    logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    start_epoch, pretrain_file = log.checkpoint_restore(model, cfg.logpath, pretrain_file=cfg.pretrain, gpu=gpu)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))
    eval_epoch(dataset.val_data_loader, model, model_fn, cfg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg = get_parser()
    # Determine whether it is distributed training
    cfg.world_size = cfg.nodes * cfg.gpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    if cfg.dist:
        print("The verification set does not support multi-card at the moment, please set nodes and gpu_per_node  to 1")
    else:
        Single_card_testing(cfg.local_rank, cfg)

