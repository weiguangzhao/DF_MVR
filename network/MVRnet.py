# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/9  下午6:53
# File Name: MVRNet.py
# IDE: PyCharm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans_f

from lib.BFM.model_basis.BFM2009.face_generate import Face3D
from tools.render import Renderer
from network.Tri_Unet import Tri_UNet
from network.Rednet import RedNet_FC
from network.Bisenet import BiSeNet
from tools.plt import save_ply
from config.config import get_parser


class MVRNet(nn.Module):
    def __init__(self, cfg):
        super(MVRNet, self).__init__()
        self.uuunet = Tri_UNet(4, 64)
        self.face3d = Face3D()
        self.rednet_fc = RedNet_FC()

        # #init render
        self.render = Renderer(check_depth=False, img_size=224)

        # # face parsing net
        self.mask_model = BiSeNet(n_classes=10)
        gpu = torch.cuda.current_device()
        map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
        pretrain_model = torch.load('pretrain/face_mask.pth', map_location=map_location)
        self.mask_model.load_state_dict(pretrain_model)

        # # fix parameter
        self.fix_module = ['mask_model']
        module_map = {'uuunet': self.uuunet, 'rednet': self.rednet_fc, 'mask_model': self.mask_model}
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False
        pass

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_lt, input_ft, input_rt):
        feature = self.uuunet(input_lt, input_ft, input_rt)
        face_para = self.rednet_fc(feature)

        # #reconstructing 3D faces of the camera coord
        recon_result = self.face3d.Reconstruction_Block(face_para)
        face_shape_lt = recon_result['face_shape_lt']
        face_shape_ft = recon_result['face_shape_ft']
        face_shape_rt = recon_result['face_shape_rt']
        face_texture = recon_result['face_texture']
        loss_para = recon_result['loss_para']
        loss_gamma = recon_result['loss_gamma']
        loss_reflect = recon_result['reflect_loss']

        # # render
        faces = self.face3d.facemodel.cell.unsqueeze(0)  # [1,F,C]
        faces = torch.repeat_interleave(faces, face_shape_lt.shape[0], dim=0)  # [B,F,C]
        render_img_lt, render_mask_lt, xyz_project_lt = self.render(face_shape_lt, face_texture, faces)
        render_img_ft, render_mask_ft, xyz_project_ft = self.render(face_shape_ft, face_texture, faces)
        render_img_rt, render_mask_rt, xyz_project_rt = self.render(face_shape_rt, face_texture, faces)

        # #rehandle
        ld_index = self.face3d.facemodel.ld_mf_index
        pre_lm101_index = self.face3d.facemodel.pre_lm101_index
        render_mask_lt = torch.repeat_interleave(render_mask_lt, 3, dim=-1)  # [B,N,C]
        render_mask_ft = torch.repeat_interleave(render_mask_ft, 3, dim=-1)  # [B,N,C]
        render_mask_rt = torch.repeat_interleave(render_mask_rt, 3, dim=-1)  # [B,N,C]

        ret = {}
        ret['render_img_lt'] = render_img_lt
        ret['render_img_ft'] = render_img_ft
        ret['render_img_rt'] = render_img_rt
        ret['render_mask_lt'] = render_mask_lt
        ret['render_mask_ft'] = render_mask_ft
        ret['render_mask_rt'] = render_mask_rt
        ret['xyz_project_lt'] = xyz_project_lt
        ret['xyz_project_ft'] = xyz_project_ft
        ret['xyz_project_rt'] = xyz_project_rt
        ret['face_shape_lt'] = face_shape_lt
        ret['face_shape_ft'] = face_shape_ft
        ret['face_shape_rt'] = face_shape_rt

        ret['ld_index'] = ld_index
        ret['pre_lm101_index'] = pre_lm101_index
        ret['face_texture'] = face_texture
        ret['loss_gamma'] = loss_gamma
        ret['loss_para'] = loss_para
        ret['loss_reflect'] = loss_reflect
        ret['mask_model'] = self.mask_model

        return ret


def model_fn(batch, model, epoch):
    # #input
    input_lt = batch['input_lt'].cuda()
    input_ft = batch['input_ft'].cuda()
    input_rt = batch['input_rt'].cuda()
    face_mask_lt = batch['face_mask_lt'].cuda()
    face_mask_ft = batch['face_mask_ft'].cuda()
    face_mask_rt = batch['face_mask_rt'].cuda()

    # # image + face maskhttps://github.com/weiguangzhao/DF_MVR/blob/main/network/MVRnet.py
    input_lt = torch.cat((input_lt, face_mask_lt), dim=1)
    input_ft = torch.cat((input_ft, face_mask_ft), dim=1)
    input_rt = torch.cat((input_rt, face_mask_rt), dim=1)
    file_name = batch['file_name']

    # # result
    ret = model(input_lt, input_ft, input_rt)

    cfg = get_parser()

    # #label
    raw_lt = batch['raw_lt'].cuda()
    raw_ft = batch['raw_ft'].cuda()
    raw_rt = batch['raw_rt'].cuda()
    kp_img_gt_lt = batch['kp_img_gt_lt'].cuda()*224.0
    kp_img_gt_ft = batch['kp_img_gt_ft'].cuda()*224.0
    kp_img_gt_rt = batch['kp_img_gt_rt'].cuda()*224.0
    weight_map_lt = batch['weight_map_lt'].permute(0, 2, 3, 1).cuda()
    weight_map_ft = batch['weight_map_ft'].permute(0, 2, 3, 1).cuda()
    weight_map_rt = batch['weight_map_rt'].permute(0, 2, 3, 1).cuda()

    kp_3d_gt = batch['kp_3d_gt'].cuda() *100

    # # landmarks loss (L2 loss)
    # set higher weights for landmarks around the mouth and nose regions
    xyz_project_lt = ret['xyz_project_lt']
    xyz_project_ft = ret['xyz_project_ft']
    xyz_project_rt = ret['xyz_project_rt']
    ld_index = ret['ld_index']
    kp_img_pred_lt = xyz_project_lt[:, ld_index, :2]
    kp_img_pred_ft = xyz_project_ft[:, ld_index, :2]
    kp_img_pred_rt = xyz_project_rt[:, ld_index, :2]

    landmark_weight = torch.cat([torch.ones([1, 28]), 20 * torch.ones([1, 3]), torch.ones([1, 5]), 20 * torch.ones([1, 12]),
                                 torch.ones([1, 12]), 20 * torch.ones([1, 8])], dim=1).cuda()
    kp_img_all_num = kp_img_gt_lt.shape[0] * kp_img_gt_lt.shape[1]
    loss_kp_img_lt = torch.sum(torch.norm((kp_img_pred_lt - kp_img_gt_lt), p=2, dim=2) * landmark_weight)/kp_img_all_num
    loss_kp_img_ft = torch.sum(torch.norm((kp_img_pred_ft - kp_img_gt_ft), p=2, dim=2) * landmark_weight)/kp_img_all_num
    loss_kp_img_rt = torch.sum(torch.norm((kp_img_pred_rt - kp_img_gt_rt), p=2, dim=2) * landmark_weight)/kp_img_all_num
    loss_kp_img = (loss_kp_img_lt + loss_kp_img_ft + loss_kp_img_rt)/3.0

    # # photo loss
    render_img_lt = ret['render_img_lt']
    render_img_ft = ret['render_img_ft']
    render_img_rt = ret['render_img_rt']
    render_mask_lt = ret['render_mask_lt']
    render_mask_ft = ret['render_mask_ft']
    render_mask_rt = ret['render_mask_rt']
    photo_loss_lt = Cal_Photo_loss(render_img_lt, raw_lt, weight_map_lt * render_mask_lt)
    photo_loss_ft = Cal_Photo_loss(render_img_ft, raw_ft, weight_map_ft * render_mask_ft)
    photo_loss_rt = Cal_Photo_loss(render_img_rt, raw_rt, weight_map_rt * render_mask_rt)
    loss_photo = (photo_loss_lt + photo_loss_ft + photo_loss_rt)/3.0

    # # mask loss
    if epoch >  cfg.mask_epoch:
        mask_model = ret['mask_model']
        loss_mask_lt = Cal_mask_loss(render_img_lt,  face_mask_lt,  mask_model)
        loss_mask_ft = Cal_mask_loss(render_img_ft,  face_mask_ft,  mask_model)
        loss_mask_rt = Cal_mask_loss(render_img_rt,  face_mask_rt,  mask_model)
        loss_mask = (loss_mask_lt + loss_mask_ft + loss_mask_rt)/3.0

    # # lm101 loss
    face_shape_ft = ret['face_shape_ft']
    pre_lm101_index = ret['pre_lm101_index']
    kp_3d_pred = face_shape_ft[:, pre_lm101_index, :]
    loss_kp_3d = calculate_lm101_loss(kp_3d_gt, kp_3d_pred)

    # # calculate the total loss
    loss_gamma = ret['loss_gamma']
    loss_para = ret['loss_para']
    loss_reflect = ret['loss_reflect']
    if epoch >  cfg.mask_epoch:
        loss = loss_kp_img * 2.0e-2 + loss_para * 3.0e-4 + 5 * loss_reflect + \
               loss_photo * 24 + 10 * loss_gamma + loss_mask * 1 + loss_kp_3d
    else:
        loss = loss_kp_img * 2.0e-2 + loss_para * 3.0e-4 + 5 * loss_reflect + \
               loss_photo * 24 + 10 * loss_gamma + loss_kp_3d

    with torch.no_grad():
        loss_kp_img = loss_kp_img * 2.0e-2
        loss_para = loss_para * 3.0e-4
        loss_photo = loss_photo * 24
        loss_gamma = loss_gamma * 10
        loss_reflect = loss_reflect * 5
        loss_kp_3d = loss_kp_3d


        visual_dict = {}
        visual_dict['loss'] = loss
        visual_dict['loss_para'] = loss_para
        visual_dict['loss_kp_img'] = loss_kp_img
        visual_dict['loss_photo'] = loss_photo
        visual_dict['loss_reflect'] = loss_reflect
        visual_dict['loss_gamma'] = loss_gamma
        visual_dict['loss_kp_3d'] = loss_kp_3d


        meter_dict = {}
        meter_dict['loss'] = (loss.item(), input_ft.shape[0])
        meter_dict['loss_para'] = (loss_para.item(), input_ft.shape[0])
        meter_dict['loss_kp_img'] = (loss_kp_img.item(), input_ft.shape[0])
        meter_dict['loss_photo'] = (loss_photo.item(), input_ft.shape[0])
        meter_dict['loss_reflect'] = (loss_reflect.item(), input_ft.shape[0])
        meter_dict['loss_gamma'] = (loss_gamma.item(), input_ft.shape[0])
        meter_dict['loss_kp_3d'] = (loss_kp_3d.item(), input_ft.shape[0])
        # meter_dict['loss_lm68'] = (loss_lm68.item(), input_ft.shape[0])

        if epoch > cfg.mask_epoch:
            loss_mask = loss_mask * 1
            visual_dict['loss_mask'] = loss_mask
            meter_dict['loss_mask'] = (loss_mask.item(), input_ft.shape[0])

        # #return ret
        ret['raw_lt'] = raw_lt
        ret['raw_ft'] = raw_ft
        ret['raw_rt'] = raw_rt
        ret['kp_img_pred_lt'] = kp_img_pred_lt
        ret['kp_img_pred_ft'] = kp_img_pred_ft
        ret['kp_img_pred_rt'] = kp_img_pred_rt
        ret['kp_img_gt_lt'] = kp_img_gt_lt
        ret['kp_img_gt_ft'] = kp_img_gt_ft
        ret['kp_img_gt_rt'] = kp_img_gt_rt
        ret['file_name'] = file_name
    return loss, ret, visual_dict, meter_dict


#  #calculate the photo loss (rgb)
def Cal_Photo_loss(render_img, ft_raw_filter, img_mask):
    img_mask = img_mask[:, :, :, 0].detach()

    # #photo loss with skin attention
    rgb_loss = torch.norm((ft_raw_filter*255.0 - render_img*255.0), p=2, dim=3)  # L2 loss
    rgb_loss = rgb_loss * img_mask/255.0
    rgb_loss = torch.sum(rgb_loss) / torch.max(torch.sum(img_mask), torch.tensor(1.0).cuda())
    return rgb_loss


#  #calculate the mask loss (rgb)
def Cal_mask_loss(render_img, raw_img_mask, mask_model):

    render_img_face = render_img.permute(0, 3, 1, 2)
    # init face parsing model
    render_img_face = trans_f.normalize(render_img_face.squeeze(0).detach().cpu(), mean=(0.50, 0.50, 0.50), std=(0.50, 0.50, 0.50))
    render_img_face = render_img_face.unsqueeze(0).cuda()
    if render_img_face.shape[0] == 1:
        render_img_face = torch.repeat_interleave(render_img_face, repeats=2, dim=0)
        render_img_mask = mask_model(render_img_face)[0]
        render_img_mask = render_img_mask[0, ...].unsqueeze(0)
    else:
        render_img_mask = mask_model(render_img_face)[0]
    cross_ent = nn.CrossEntropyLoss().cuda()
    loss_mask = cross_ent(render_img_mask, raw_img_mask.squeeze(1).long())

    # fig, ax = plt.subplots()
    # plt.axis('off')
    # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # rend_result = render_img_mask.permute(0, 2, 3, 1).max(-1)[1]
    # plt.imshow(rend_result.permute(1,2,0).detach().cpu().numpy())
    # plt.savefig('rdm_l.png', dpi=300)
    # print('ss')

    return loss_mask


def calculate_lm101_loss(target, output):
    target = target.view(-1, 101, 3)
    output = output.view(-1, 101, 3)
    pose_align_idx = [47, 50, 53, 56, 41, 79, 85]
    output = trans_point(target, output, pose_align_idx)
    loss_temp = torch.norm(output - target, dim=2).sum(dim=1)
    loss = loss_temp.sum()
    loss = loss / (target.shape[0] * 101)
    return loss


def trans_point(gtpoint, infpoint, pose_align_idx):
    inpoint = gtpoint[:, pose_align_idx, :]
    outpoint = infpoint[:, pose_align_idx, :]
    mean_inpoint = - inpoint.mean(dim=1, keepdim=True)
    mean_outpoint = - outpoint.mean(dim=1, keepdim=True)

    inpoint += mean_inpoint
    outpoint += mean_outpoint

    for i, gt, gt0, pt, pt0 in zip(range(inpoint.shape[0]), inpoint, mean_inpoint, outpoint, mean_outpoint):
        gt = gt.T
        pt = pt.T
        gt0 = gt0.T
        pt0 = pt0.T
        covariance_matrix = torch.mm(gt, pt.T)
        U, S, V = torch.svd(covariance_matrix)
        R = U.mm(V.T)

        if torch.det(R) < 0:
            R[:, 2] *= -1

        rms_gt = torch.mean(torch.norm(gt, dim=0) ** 2) ** 0.5
        rms_pt = torch.mean(torch.norm(pt, dim=0) ** 2) ** 0.5

        s = (rms_gt / rms_pt)
        P = s * torch.eye(3).cuda().mm(R).cuda()

        t_final = P.mm(pt0) - gt0
        P = torch.cat([P, t_final], dim=1)
        tempPoint = torch.cat([infpoint[i], torch.ones([infpoint[i].shape[0], 1]).cuda()], dim=1)
        infpoint[i] = P.mm(tempPoint.T).T

    return infpoint
