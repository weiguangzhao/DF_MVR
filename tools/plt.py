# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/9  下午11:23
# File Name: plt.py
# IDE: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def save_pcd(points, save_pcd_path):
    pcd_header = '''\
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z intensity
    SIZE 4 4 4 4 
    TYPE F F F F 
    COUNT 1 1 1 1 
    WIDTH {}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {}
    DATA ascii
    '''
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z= points[i]
        lines.append('{:.6f} {:.6f} {:.6f} {}'.format(x, y, z, i))
    with open(save_pcd_path, 'w') as f:
        f.write(pcd_header.format(n, n))
        f.write('\n'.join(lines))


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

# v: point c:color f:face
def save_obj(v, c, f, save_obj_path):
    obj_header = '''\
        nothing
        point: {}
        face: {}\n'''
    lines = []
    for i in range(v.shape[0]):
        x, y, z = v[i]
        r, g, b = c[i]
        lines.append('v {:.8f} {:.8f} {:.8f} {:d} {:d} {:d}'.format(x, y, z, int(r), int(g), int(b)))
    for i in range(f.shape[0]):
        f_1, f_2, f_3 = f[i]
        lines.append('f {:d} {:d} {:d}'.format(int(f_1), int(f_2), int(f_3)))
    pass
    with open(save_obj_path, 'w') as file:
        file.write(obj_header.format(v.shape[0], f.shape[0]))
        file.write('\n'.join(lines))


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
    pass
