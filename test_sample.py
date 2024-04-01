# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2024/1/10  下午1:05
# File Name: test_sample.py
# IDE: PyCharm

import os
import glob
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import tools.log as log
from tools.plt import save_obj
from network.Bisenet import BiSeNet
from config.config_test import get_parser
from lib.BFM.model_basis.BFM2009.face_generate import Face3D
from dataset.pixel_face.dataset_preprocess import get_face_mask

def save_pred(preds, save_path):
    face_shape_ft = preds['face_shape_ft']
    face_texture = preds['face_texture']
    face_model = Face3D()

    # # save 3d obj
    face_obj = face_model.facemodel.cell
    vertices_img_obj = face_shape_ft[0, ...].detach().cpu().numpy()
    vertices_img_color_obj = np.ones_like(vertices_img_obj)*127.5
    # vertices_img_color_obj = face_texture[0, :, :].detach().cpu().numpy() * 255.0
    save_obj(vertices_img_obj, vertices_img_color_obj, face_obj+1, save_path)
    pass

if __name__ == '__main__':

    result_path_obj = 'samples_results/' + 'obj/'
    os.makedirs(result_path_obj, exist_ok=True)  # create dir

    # # fix seed for debug
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)  # cpu
    torch.cuda.manual_seed(22)
    cfg = get_parser()

    # #INIT
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # #normalized rgb to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # ####get face mask
    mask_model = BiSeNet(10)
    mask_model.cuda()
    gpu = torch.cuda.current_device()
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    mask_model.load_state_dict(torch.load('pretrain/face_mask.pth', map_location=map_location))
    mask_model.eval()

    # #loading DF_MVR model
    print('=> creating model ...')
    from network.MVRnet import MVRNet as net
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg)
    model = model.to(gpu)
    model.eval()
    print('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    start_epoch, pretrain_file = log.checkpoint_restore(model, cfg.logpath, pretrain_file=cfg.pretrain, gpu=gpu)
    print('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0 else 'Start from epoch {}'.format(start_epoch))

    files_list = glob.glob('samples/1/*.png')
    for f_i, f_path in enumerate(files_list):
        f_name = f_path.split('/')[-1][:-5]
        fp_0 = 'samples/0/{}0.png'.format(f_name)
        fp_1 = 'samples/1/{}1.png'.format(f_name)
        fp_2 = 'samples/2/{}2.png'.format(f_name)
        image_0 = np.array(Image.open(fp_0))
        image_1 = np.array(Image.open(fp_1))
        image_2 = np.array(Image.open(fp_2))
        img_norm_0 = transform(image_0.copy())
        img_norm_1 = transform(image_1.copy())
        img_norm_2 = transform(image_2.copy())
        face_mask_0, _ = get_face_mask(img_norm_0, mask_model)
        face_mask_1, _ = get_face_mask(img_norm_1, mask_model)
        face_mask_2, _ = get_face_mask(img_norm_2, mask_model)
        face_mask_0 = torch.from_numpy(face_mask_0[np.newaxis, np.newaxis, ...]).type(torch.float32)
        face_mask_1 = torch.from_numpy(face_mask_1[np.newaxis, np.newaxis, ...]).type(torch.float32)
        face_mask_2 = torch.from_numpy(face_mask_2[np.newaxis, np.newaxis, ...]).type(torch.float32)
        input_0 = torch.cat((img_norm_0.unsqueeze(0), face_mask_0), dim=1).cuda()
        input_1 = torch.cat((img_norm_1.unsqueeze(0), face_mask_1), dim=1).cuda()
        input_2 = torch.cat((img_norm_2.unsqueeze(0), face_mask_2), dim=1).cuda()
        ret = model(input_0, input_1, input_2)
        save_path = result_path_obj + f_name + '.obj'
        save_pred(ret, save_path)
        print('complete {} / {}, save to {}'.format(f_i, len(files_list), save_path))