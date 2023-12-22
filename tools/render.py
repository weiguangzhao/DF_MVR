# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/7/8  下午4:42
# File Name: render.py
# IDE: PyCharm

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.lighting import AmbientLights, PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.renderer import MeshRenderer, MeshRendererWithFragments
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    HardPhongShader,
    SoftPhongShader,
)

from collections import namedtuple

ShaderTest = namedtuple("ShaderTest", ["shader", "reference_name", "debug_name"])


class Renderer(nn.Module):
    def __init__(self, check_depth=False, img_size=224):
        super(Renderer, self).__init__()

        self.img_size = img_size
        # # get the current cuda
        cuda_device_num = torch.cuda.current_device()
        cuda_device = 'cuda:' + str(cuda_device_num)

        # Init rasterizer settings
        R, T = look_at_view_transform()
        self.cameras = FoVPerspectiveCameras(device=cuda_device, R=R, T=T, znear=0.01, zfar=50.0, fov=12.593637)

        # Init shader settings
        materials = Materials(device=cuda_device)
        self.lights = PointLights(device=cuda_device, ambient_color=((1.0, 1.0, 1.0),))
        self.lights.location = torch.tensor([0.0, 0.0, -1e5], device=cuda_device)[None]
        raster_settings = RasterizationSettings(
            image_size=self.img_size, blur_radius=0.0, faces_per_pixel=1
        )
        rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

        phong_shader = SoftPhongShader(
            lights=self.lights,
            cameras=self.cameras,
            materials=materials,
            blend_params=blend_params,
        )

        if check_depth:
            self.phong_renderer = MeshRendererWithFragments(
                rasterizer=rasterizer, shader=phong_shader
            )
        else:
            self.phong_renderer = MeshRenderer(
                rasterizer=rasterizer, shader=phong_shader
            )
        pass

    def forward(self, xyz, rgb, cell):

        # Init mesh
        textures = TexturesVertex(verts_features=rgb)
        face_mesh = Meshes(verts=xyz, faces=cell, textures=textures)

        # #render img
        images = self.phong_renderer(face_mesh, lights=self.lights)

        # # project
        size = torch.tensor([self.img_size, self.img_size]).unsqueeze(0).cuda()
        # img_size = torch.repeat_interleave(size, repeats=xyz.shape[0], dim=0).cuda()
        xyz_project = self.cameras.transform_points_screen(xyz, image_size=size)

        render_img = images[..., :3]
        render_mask = images[..., 3].unsqueeze(-1)
        return render_img, render_mask, xyz_project


# # #for test
# def test_simple_sphere(xyz, rgb, cell, elevated_camera=False, check_depth=False):
#     """
#     Test output of phong and gouraud shading matches a reference image using
#     the default values for the light sources.
#     Args:
#         elevated_camera: Defines whether the camera observing the scene should
#                        have an elevation of 45 degrees.
#     """
#
#     # Init rasterizer settings
#     if elevated_camera:
#         # Elevated and rotated camera
#         R, T = look_at_view_transform(dist=2.7, elev=45.0, azim=45.0)
#         postfix = "_elevated_"
#         # If y axis is up, the spot of light should
#         # be on the bottom left of the sphere.
#     else:
#         # No elevation or azimuth rotation
#         R, T = look_at_view_transform(3, 0.0, 0.0)
#         postfix = "_"
#     cameras = FoVPerspectiveCameras(device=xyz.device, R=R, T=T, znear=0.01, zfar=50.0)
#
#     # Init mesh
#     textures = TexturesVertex(verts_features=rgb)
#     sphere_mesh = Meshes(verts=xyz, faces=cell, textures=textures)
#     size = torch.tensor([224, 224]).cuda().unsqueeze(0)
#     img_size = torch.repeat_interleave(size, repeats=1, dim=0).cuda()
#     xyz_project = cameras.transform_points_screen(xyz, image_size=img_size)
#     xyz_project = xyz_project.detach().cpu().numpy()
#
#     # Init shader settings
#     materials = Materials(device=xyz.device)
#     lights = PointLights(device=xyz.device)
#     lights.location = torch.tensor([0.0, 0.0, 1e5], device=xyz.device)[None]
#
#     raster_settings = RasterizationSettings(
#         image_size=224, blur_radius=0.0, faces_per_pixel=1
#     )
#     rasterizer = MeshRasterizer(
#         cameras=cameras, raster_settings=raster_settings
#     )
#     blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))
#
#     # shader = HardPhongShader(
#     #     lights=lights,
#     #     cameras=cameras,
#     #     materials=materials,
#     #     blend_params=blend_params,
#     # )
#     # if check_depth:
#     #     renderer = MeshRendererWithFragments(
#     #         rasterizer=rasterizer, shader=shader
#     #     )
#     #     images, fragments = renderer(sphere_mesh)
#     # else:
#     #     renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
#     #     images = renderer(sphere_mesh)
#     # rgb = images[0, ..., :3].squeeze().detach().cpu()
#     # rgb_plt = rgb.numpy()
#     # plt.imshow(rgb_plt)
#     # plt.show()
#
#     ########################################################
#     # Move the light to the +z axis in world space so it is
#     # behind the sphere. Note that +Z is in, +Y up,
#     # +X left for both world and camera space.
#     ########################################################
#     lights.location[..., 2] = 1e5
#     phong_shader = SoftPhongShader(
#         lights=lights,
#         cameras=cameras,
#         materials=materials,
#         blend_params=blend_params,
#     )
#     if check_depth:
#         phong_renderer = MeshRendererWithFragments(
#             rasterizer=rasterizer, shader=phong_shader
#         )
#         images, fragments = phong_renderer(sphere_mesh, lights=lights)
#
#     else:
#         phong_renderer = MeshRenderer(
#             rasterizer=rasterizer, shader=phong_shader
#         )
#         images = phong_renderer(sphere_mesh, lights=lights)
#
#     render_img = images[..., :3]
#     render_mask = images[..., 3]
#
#     # #plt
#     index = np.load('../lib/BFM/model_basis/BFM2017/landmarks/ld_mf_index.npy')
#     img_plt = render_img[0, ...].detach().cpu().numpy()
#     plt.imshow(img_plt)
#     plt.scatter(xyz_project[0, index, 0], xyz_project[0, index, 1], c='r', s=2)
#     plt.show()
#     return render_img, render_mask
#
#
# torch.cuda.set_device(4)
# filename = '../lib/BFM/model_basis/BFM2017/model2017-1_face12_nomouth.h5'
# f = h5py.File(filename, "r")
# color = f['color']['model']['mean']
# color = np.array(color).reshape(1, -1, 3)
# xyz = f['shape']['model']['mean']
# xyz = np.array(xyz).reshape(1, -1, 3)
# exp = f['expression']['model']['mean']
# exp = np.array(exp).reshape(1, -1, 3)
# mean_shape = xyz + exp
# mean_color = color
# cell = f['shape']['representer']['cells']
# cell = np.array(cell).transpose(1, 0)
#
# mean_color = torch.tensor(mean_color, requires_grad=True).cuda()
# mean_shape = torch.tensor(mean_shape, requires_grad=True).cuda() / 100.0
# cell = torch.from_numpy(cell).cuda()
# cell = torch.unsqueeze(cell, 0).type(torch.int64)
# test_simple_sphere(mean_shape, mean_color, cell)


# def test_simple_sphere_batched():
#     """
#     Test a mesh with vertex textures can be extended to form a batch, and
#     is rendered correctly with Phong, Gouraud and Flat Shaders with batched
#     lighting and hard and soft blending.
#     """
#     batch_size = 5
#     device = torch.device("cuda:0")
#
#     # Init mesh with vertex textures.
#     sphere_meshes = ico_sphere(5, device).extend(batch_size)
#     verts_padded = sphere_meshes.verts_padded()
#     faces_padded = sphere_meshes.faces_padded()
#     feats = torch.ones_like(verts_padded, device=device)
#     textures = TexturesVertex(verts_features=feats)
#     sphere_meshes = Meshes(
#         verts=verts_padded, faces=faces_padded, textures=textures
#     )
#
#     # Init rasterizer settings
#     dist = torch.tensor([2.7]).repeat(batch_size).to(device)
#     elev = torch.zeros_like(dist)
#     azim = torch.zeros_like(dist)
#     R, T = look_at_view_transform(dist, elev, azim)
#     cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
#     raster_settings = RasterizationSettings(
#         image_size=512, blur_radius=0.0, faces_per_pixel=4
#     )
#
#     # Init shader settings
#     materials = Materials(device=device)
#     lights_location = torch.tensor([0.0, 0.0, +2.0], device=device)
#     lights_location = lights_location[None].expand(batch_size, -1)
#     lights = PointLights(device=device, location=lights_location)
#     blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))
#
#     # Init renderer
#     rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
#     shader_tests = [
#         ShaderTest(HardPhongShader, "phong", "hard_phong"),
#         ShaderTest(SoftPhongShader, "phong", "soft_phong")
#     ]
#     for test in shader_tests:
#         reference_name = test.reference_name
#         debug_name = test.debug_name
#         shader = test.shader(
#             lights=lights,
#             cameras=cameras,
#             materials=materials,
#             blend_params=blend_params,
#         )
#         renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
#         images = renderer(sphere_meshes)
#
# test_simple_sphere_batched()