'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://ringnet.is.tue.mpg.de/license). 
Any use not explicitly granted by the LICENSE is prohibited.
Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.
More information about the NoW Challenge is available at https://ringnet.is.tue.mpg.de/challenge.
For comments or questions, please email us at ringnet@tue.mpg.de
'''
import os
import time
from glob import glob
import sys
sys.path.append('lib/now_evaluation/')

import numpy as np
import scan2mesh_computations as s2m_opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from psbody.mesh import Mesh
import chumpy as ch
# # save ply file
def save_ply(xyz, rgb, filename):
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        \n
        '''
    xyz = np.hstack([xyz.reshape(-1, 3), rgb])
    np.savetxt(filename, xyz, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(xyz)))
        f.write(old)
    pass


def get_ptcloud_img(xyz, rgb):
    fig = plt.figure(figsize=(5, 5))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax = Axes3D(fig)
    ax.view_init(90, -90)
    # ax.axis('off')

    # plot point
    # max, min = np.max(xyz), np.min(xyz)
    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)
    ax.set_xbound(min_x, max_x)
    ax.set_ybound(min_y, max_y)
    ax.set_zbound(min_z, max_z)
    ax.scatter(x, y, z, zdir='z', c=rgb, marker='.', s=20)
    # plt.savefig('test.png')
    plt.show()
    pass\


def get_obj_info(obj_dir):
    # handle obj_data
    with open(obj_dir) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                # faces.append((float(strs[1]), float(strs[2]), float(strs[3]))) # pixel_face
                faces.append((int(strs[1].split('/')[0]), int(strs[2].split('/')[0]), int(strs[3].split('/')[0]))) # micc
        obj_point = np.array(points)
        obj_face = np.array(faces)
        obj_color = np.ones_like(obj_point) * 125.0
    return obj_point, obj_color, obj_face


def compute_error_metric(gt_point, gt_face, gt_al, pre_point, pre_face, pre_al):
    distances = s2m_opt.compute_errors(gt_point, gt_face, gt_al, pre_point, pre_face, pre_al)
    return np.stack(distances)


def compute_rmse(pre_point, pre_face, fn):
    print(fn)
    GT_obj_path = os.path.join('datasets/pixel_face/val_obj', fn + '.obj')
    gt_point, gt_color, gt_face = get_obj_info(GT_obj_path)
    #gt_face = np.load('dataset/bo/gt/' + fn + '_face.npy')
    #gt_point = np.load('dataset/bo/gt/' + fn + '_vert.npy')
    #gt_al = np.load('dataset/bo/gt/' + fn + '_3dld.npy')
    if np.min(gt_face) != 0:
        gt_face = gt_face - 1
    if np.min(pre_face) != 0:
        pre_face = pre_face - 1
    gt_al_index = np.load('lib/now_evaluation/gt_al_index.npy')
    pre_al_index = np.load('lib/now_evaluation/pre_al_index.npy')
    gt_al = gt_point[gt_al_index, :]
    pre_al = pre_point[pre_al_index, :]

    rmse=compute_error_metric(gt_point, gt_face, gt_al, pre_point, pre_face, pre_al)
    rmse = np.array(rmse)
    rmse = np.mean(rmse)*100      # pixel_face
    print('fn: ', fn)
    return rmse

