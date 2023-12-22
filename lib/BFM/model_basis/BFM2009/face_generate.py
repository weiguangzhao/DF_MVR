# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/9  下午7:01
# File Name: face_generate.py
# IDE: PyCharm

import torch
import math as m
import numpy as np
import h5py
from scipy.io import loadmat
from tools.plt import get_ptcloud_img
import torch.nn as nn


###############################################################################################
# Reconstruct 3D face based on output coefficients and facemodel 2009
###############################################################################################


# BFM 3D face model
class BFM():
    def __init__(self):
        # #BFM2009
        # avg_shape, base_shape = self.load_Basel_basic('shape')
        # avg_exp, base_exp = self.load_Basel_basic('exp')

        # #BFM2009
        filename = 'lib/BFM/model_basis/BFM2009/BFM_model_front.mat'
        model = loadmat(filename)
        mean_shape, mean_color, xyz_basis, rgb_basis, exp_basis, cell, ld_index = self.load_mat(filename)
        self.mean_shape = torch.tensor(mean_shape.astype(np.float32)).cuda()  # mean face shape. [3*N,1]
        self.mean_color = torch.tensor(mean_color.astype(np.float32)).cuda()  # mean face shape. [3*N,1]
        self.xyz_basis = torch.tensor(xyz_basis.astype(np.float32)).cuda()  # identity basis. [3*N,199]
        self.exp_basis = torch.tensor(exp_basis.astype(np.float32)).cuda()  # expression basis. [3*N,100]
        self.rgb_basis = torch.tensor(rgb_basis.astype(np.float32)).cuda()  # color basis. [3*N,199]
        self.cell = torch.tensor(cell.astype(np.float32)).cuda()  # Tri  [21*N,3]

        # #load the landmarks of the face_generate or face_detect
        self.ld_mf_index = ld_index.astype(np.int32)
        # face indices for each vertex that lies in. starts from 1. [N,8]
        self.point_buf = torch.tensor(model['point_buf']).cuda() - 1

        self.d3_skin_mask = torch.squeeze(torch.tensor(model['skinmask']).cuda())

        # #load 101 lm_index
        self.pre_lm101_index = np.load('lib/BFM/model_basis/BFM2009/index/pre_index_101.npy')


    def load_mat(self, filename):
        model = loadmat(filename)
        mean_shape = model['meanshape']
        mean_color = model['meantex']
        cell = model['tri'].astype(np.int32) - 1
        xyz_basis = model['idBase']
        exp_basis = model['exBase']
        rgb_basis = model['texBase']
        ld_index = model['keypoints'] -1

        mean_shape = np.squeeze(mean_shape)
        mean_color = np.squeeze(mean_color)
        ld_index = np.squeeze(ld_index)

        return mean_shape, mean_color, xyz_basis, rgb_basis, exp_basis, cell, ld_index

# Analytic 3D face
class Face3D():
    def __init__(self):
        facemodel = BFM()
        self.facemodel = facemodel

    # analytic 3D face reconstructions with face_para
    def Reconstruction_Block(self, face_para):
        face_id = face_para['face_id']
        face_exp = face_para['face_exp']
        face_color = face_para['face_color']
        face_gamma = face_para['face_gamma']

        lt_rotate = face_para['face_rotate_lt']
        lt_offset = face_para['face_offset_xyz_lt']
        ft_rotate = face_para['face_rotate_ft']
        ft_offset = face_para['face_offset_xyz_ft']
        rt_rotate = face_para['face_rotate_rt']
        rt_offset = face_para['face_offset_xyz_rt']

        face_shape = self.Shape_formation_block(face_id, face_exp, self.facemodel)
        face_texture = self.Texture_formation_block(face_color, self.facemodel)

        # # do rigid transformation for face shape using predicted rotation and translation
        rotation_matrix = self.Compute_rotation_matrix(lt_rotate)
        face_shape_lt = self.Rigid_transform_block(face_shape, rotation_matrix, lt_offset)
        rotation_matrix = self.Compute_rotation_matrix(ft_rotate)
        face_shape_ft = self.Rigid_transform_block(face_shape, rotation_matrix, ft_offset)
        rotation_matrix = self.Compute_rotation_matrix(rt_rotate)
        face_shape_rt = self.Rigid_transform_block(face_shape, rotation_matrix, rt_offset)

        # #loss of para meter to prevent generating false face
        id_weight, exp_weight, color_weight = 1, 0.8, 0.017
        loss_para = id_weight * self.calculate_para_loss(face_id) + exp_weight * self.calculate_para_loss(face_exp) + \
                    color_weight * self.calculate_para_loss(face_color)

        # for gamma loss
        norm_shape = self.Compute_norm(face_shape, self.facemodel)
        norm_shape_rotate = torch.matmul(norm_shape, rotation_matrix)
        face_texture = self.Illumination_block(face_texture, norm_shape_rotate, face_gamma)
        loss_gamma = self.Gamma_loss(face_gamma)

        # # for reflect loss
        reflect_loss = self.Reflectance_loss(face_texture, self.facemodel.d3_skin_mask)

        # #for render (0-255) to (0-1)
        face_texture = face_texture / 255.0

        recon_result = {}
        recon_result['face_shape_lt'] = face_shape_lt
        recon_result['face_shape_ft'] = face_shape_ft
        recon_result['face_shape_rt'] = face_shape_rt
        recon_result['face_texture'] = face_texture
        recon_result['loss_para'] = loss_para
        recon_result['loss_gamma'] = loss_gamma
        recon_result['reflect_loss'] = reflect_loss

        return recon_result

    def Shape_formation_block(self, face_id, face_exp, facemodel):
        face_shape = torch.einsum('ij,aj->ai', facemodel.xyz_basis, face_id) + \
                     torch.einsum('ij,aj->ai', facemodel.exp_basis, face_exp) + facemodel.mean_shape

        # reshape face shape to [batchsize,N,3]
        face_shape = torch.reshape(face_shape, (face_shape.shape[0], -1, 3))
        # re-centering the face shape with mean shape
        face_shape = face_shape - torch.reshape(torch.mean(torch.reshape(facemodel.mean_shape, [-1, 3]), 0), (1, 1, 3))
        return face_shape

    def Texture_formation_block(self, face_color, facemodel):
        face_texture = torch.einsum('ij,aj->ai', facemodel.rgb_basis, face_color) + facemodel.mean_color

        # reshape face shape to [batchsize,N,3]
        face_texture = torch.reshape(face_texture, (face_texture.shape[0], -1, 3))
        return face_texture

    def Compute_rotation_matrix(self, angles):
        n_data = angles.shape[0]
        # compute rotation matrix for X-axis, Y-axis, Z-axis respectively
        rotation_X = torch.cat([torch.ones([n_data, 1]).cuda(),
                                torch.zeros([n_data, 3]).cuda(),
                                torch.reshape(torch.cos(angles[:, 0]), [n_data, 1]),
                                -torch.reshape(torch.sin(angles[:, 0]), [n_data, 1]),
                                torch.zeros([n_data, 1]).cuda(),
                                torch.reshape(torch.sin(angles[:, 0]), [n_data, 1]),
                                torch.reshape(torch.cos(angles[:, 0]), [n_data, 1])], 1)

        rotation_Y = torch.cat([torch.reshape(torch.cos(angles[:, 1]), [n_data, 1]),
                                torch.zeros([n_data, 1]).cuda(),
                                torch.reshape(torch.sin(angles[:, 1]), [n_data, 1]),
                                torch.zeros([n_data, 1]).cuda(),
                                torch.ones([n_data, 1]).cuda(),
                                torch.zeros([n_data, 1]).cuda(),
                                -torch.reshape(torch.sin(angles[:, 1]), [n_data, 1]),
                                torch.zeros([n_data, 1]).cuda(),
                                torch.reshape(torch.cos(angles[:, 1]), [n_data, 1])], 1)

        rotation_Z = torch.cat([torch.reshape(torch.cos(angles[:, 2]), [n_data, 1]),
                                -torch.reshape(torch.sin(angles[:, 2]), [n_data, 1]),
                                torch.zeros([n_data, 1]).cuda(),
                                torch.reshape(torch.sin(angles[:, 2]), [n_data, 1]),
                                torch.reshape(torch.cos(angles[:, 2]), [n_data, 1]),
                                torch.zeros([n_data, 3]).cuda(),
                                torch.ones([n_data, 1]).cuda()], 1)

        rotation_X = torch.reshape(rotation_X, [n_data, 3, 3])
        rotation_Y = torch.reshape(rotation_Y, [n_data, 3, 3])
        rotation_Z = torch.reshape(rotation_Z, [n_data, 3, 3])

        # R = RzRyRx
        rotation = torch.matmul(torch.matmul(rotation_Z, rotation_Y), rotation_X)
        rotation = rotation.permute(0, 2, 1)
        return rotation

    def Rigid_transform_block(self, face_shape, rotation_matrix, face_offset):
        # do rigid transformation for 3D face shape
        face_shape_r = torch.matmul(face_shape, rotation_matrix)
        face_shape_t = face_shape_r + torch.reshape(face_offset, [face_shape.shape[0], 1, 3])
        return face_shape_t

    def calculate_para_loss(self, para):
        batch_size = float(para.shape[0])
        para = torch.pow(para, 2)
        para_loss = torch.sum(para) / batch_size
        return para_loss

    def Compute_norm(self, face_shape, facemodel):
        shape = face_shape
        face_id = facemodel.cell
        point_id = facemodel.point_buf

        # face_id and point_id index starts from 1
        face_id = face_id.type(torch.long)
        point_id = point_id.type(torch.long)

        # compute normal for each face
        v1 = shape[:, face_id[:, 0], :]
        v2 = shape[:, face_id[:, 1], :]
        v3 = shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2)

        face_norm = torch.nn.functional.normalize(face_norm, dim=2)  # normalized face_norm first
        face_norm = torch.cat((face_norm, torch.zeros([face_shape.shape[0], 1, 3]).cuda()), dim=1)

        # compute normal for each vertex using one-ring neighborhood
        v_norm = torch.sum(face_norm[:, point_id, :], dim=2)
        v_norm = torch.nn.functional.normalize(v_norm, dim=2)

        return v_norm

    def Illumination_block(self, face_texture, norm_r, gamma):
        n_data = gamma.shape[0]
        n_point = norm_r.shape[1]
        gamma = torch.reshape(gamma, [n_data, 3, 9])
        # set initial lighting with an ambient lighting
        init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0]).cuda()
        gamma = gamma + torch.reshape(init_lit, [1, 1, 9])

        # compute vertex color using SH function approximation
        a0 = m.pi
        a1 = 2 * m.pi / np.sqrt(3.0)
        a2 = 2 * m.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(m.pi * 4)
        c1 = np.sqrt(3.0) / np.sqrt(4 * m.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * m.pi)

        Y = torch.cat([a0 * c0 * torch.ones([n_data, n_point, 1]).cuda(),
                       torch.unsqueeze(-a1 * c1 * norm_r[:, :, 1], 2),
                       torch.unsqueeze(a1 * c1 * norm_r[:, :, 2], 2),
                       torch.unsqueeze(-a1 * c1 * norm_r[:, :, 0], 2),
                       torch.unsqueeze(a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 1], 2),
                       torch.unsqueeze(-a2 * c2 * norm_r[:, :, 1] * norm_r[:, :, 2], 2),
                       torch.unsqueeze(a2 * c2 * 0.5 / np.sqrt(3.0) * (3 * torch.square(norm_r[:, :, 2]) - 1), 2),
                       torch.unsqueeze(-a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 2], 2),
                       torch.unsqueeze(a2 * c2 * 0.5 * (torch.square(norm_r[:, :, 0]) - torch.square(norm_r[:, :, 1])),
                                       2)], dim=2)

        color_r = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 0, :], 2)), dim=2)
        color_g = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 1, :], 2)), dim=2)
        color_b = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 2, :], 2)), dim=2)

        # [batchsize,N,3] vertex color in RGB order
        face_color = torch.stack(
            [color_r * face_texture[:, :, 0], color_g * face_texture[:, :, 1], color_b * face_texture[:, :, 2]], dim=2)

        return face_color

    # gamma regularization to ensure a nearly-monochromatic light
    def Gamma_loss(self, gamma):
        gamma = torch.reshape(gamma, [-1, 3, 9])
        gamma_mean = torch.mean(gamma, dim=1, keepdim=True)

        gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

        return gamma_loss

    # albedo regularization to ensure an uniform skin albedo
    def Reflectance_loss(self, face_texture, d3_skin_mask):
        skin_mask = torch.reshape(d3_skin_mask, [1, d3_skin_mask.shape[0], 1])

        texture_mean = torch.sum(face_texture * skin_mask, dim=1) / torch.sum(skin_mask)
        texture_mean = texture_mean.unsqueeze(1)

        # minimize texture variance for pre-defined skin region
        reflectance_loss = torch.sum(torch.square((face_texture - texture_mean) * skin_mask / 255.0)) / \
                           (face_texture.shape[0] * torch.sum(skin_mask))

        return reflectance_loss
