# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/12  下午6:49
# File Name: dataset_preprocess.py
# IDE: PyCharm

import os
import cv2
import glob
import torch
import random
import numpy as np
import SharedArray as SA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import PIL.ImageEnhance as ImageEnhance

from mtcnn import MTCNN
import face_alignment

from network.Bisenet import BiSeNet


# ###################################create shm####################################################
def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x

class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, sharpness =None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]
        if not sharpness is None and sharpness>0:
            self.sharpness = [max(1-sharpness, 0), 1+sharpness]

    def __call__(self, im_f, im_l, im_r):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        r_sharpness = random.uniform(self.sharpness[0], self.sharpness[1])
        im_f = ImageEnhance.Brightness(im_f).enhance(r_brightness)
        im_f = ImageEnhance.Contrast(im_f).enhance(r_contrast)
        im_f = ImageEnhance.Color(im_f).enhance(r_saturation)
        im_f = ImageEnhance.Sharpness(im_f).enhance(r_sharpness)

        im_l = ImageEnhance.Brightness(im_l).enhance(r_brightness)
        im_l = ImageEnhance.Contrast(im_l).enhance(r_contrast)
        im_l = ImageEnhance.Color(im_l).enhance(r_saturation)
        im_l = ImageEnhance.Sharpness(im_l).enhance(r_sharpness)

        im_r = ImageEnhance.Brightness(im_r).enhance(r_brightness)
        im_r = ImageEnhance.Contrast(im_r).enhance(r_contrast)
        im_r = ImageEnhance.Color(im_r).enhance(r_saturation)
        im_r = ImageEnhance.Sharpness(im_r).enhance(r_sharpness)
        return im_f, im_l, im_r


def create_shared_memory(List_name):
    for i, list_name in enumerate(List_name):
        fn = list_name.split('/')[3] + list_name.split('/')[4] + list_name.split('/')[6]
        if not os.path.exists("/dev/shm/{}_weight_map".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            if list_name.split('/')[6] == '1':
                kp_3d_gt = np.load("dataset/pixel_face/npy/{}_kp_3d_gt.npy".format(fn)).copy()
                sa_create("shm://{}_kp_3d_gt".format(fn), kp_3d_gt)

            img_norm = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn)).copy()
            img_raw = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn)).copy()
            kp_img_gt = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn)).copy()
            face_mask = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn)).copy()
            weight_map = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn)).copy()

            sa_create("shm://{}_img_norm".format(fn), img_norm)
            sa_create("shm://{}_img_raw".format(fn), img_raw)
            sa_create("shm://{}_kp_img_gt".format(fn), kp_img_gt)
            sa_create("shm://{}_face_mask".format(fn), face_mask)
            sa_create("shm://{}_weight_map".format(fn), weight_map)


def delete_shared_memory(List_name):
    for list_name in List_name:
        fn = list_name.split('/')[3] + list_name.split('/')[4] + list_name.split('/')[6]  # get shm name
        if os.path.exists("/dev/shm/{}_img_norm".format(fn)):
            SA.delete("shm://{}_img_norm".format(fn))
            SA.delete("shm://{}_img_raw".format(fn))
            SA.delete("shm://{}_kp_img_gt".format(fn))
            SA.delete("shm://{}_face_mask".format(fn))
            SA.delete("shm://{}_weight_map".format(fn))
            if list_name.split('/')[6] == '1':
                SA.delete("shm://{}_kp_3d_gt".format(fn))
            print('delete ', fn)


def create_npy(List_name, mask_model):
    for i, list_name in enumerate(List_name):
        fn = list_name.split('/')[3] + list_name.split('/')[4] + list_name.split('/')[6]
        if not os.path.exists("dataset/pixel_face/npy/{}_weight_map.npy".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            if list_name.split('/')[6] == '1':
                img_norm, img_raw, kp_img_gt, face_mask, weight_map, kp_3d_gt = Data_preprocess(list_name, mask_model)
                np.save("dataset/pixel_face/npy/{}_kp_3d_gt.npy".format(fn), kp_3d_gt)
            else:
                img_norm, img_raw, kp_img_gt, face_mask, weight_map = Data_preprocess(list_name, mask_model)

            np.save("dataset/pixel_face/npy/{}_img_norm.npy".format(fn), img_norm)
            np.save("dataset/pixel_face/npy/{}_img_raw.npy".format(fn), img_raw)
            np.save("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn), kp_img_gt)
            np.save("dataset/pixel_face/npy/{}_face_mask.npy".format(fn), face_mask)
            np.save("dataset/pixel_face/npy/{}_weight_map.npy".format(fn), weight_map)


# ###################################image process####################################################
# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


# # image process
def load_lm3d():
    Lm3D = loadmat('/lib/BFM/model_basis/BFM2009/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']
    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
    return Lm3D


def load_img_and_lm(img_path, detector):
    image = Image.open(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = RotateClockWise90(img)
    img = cv2.flip(img, 1)  # #according to obj to flip
    # plt.imshow(img)  # check img
    # plt.show()

    face = detector.detect_faces(img)[0]
    left_eye = face["keypoints"]["left_eye"]
    right_eye = face["keypoints"]["right_eye"]
    nose = face["keypoints"]["nose"]
    mouth_left = face["keypoints"]["mouth_left"]
    mouth_right = face["keypoints"]["mouth_right"]
    lm = np.array([[left_eye[0], left_eye[1]],
                   [right_eye[0], right_eye[1]],
                   [nose[0], nose[1]],
                   [mouth_left[0], mouth_left[1]],
                   [mouth_right[0], mouth_right[1]]])

    image = image.rotate(-90, expand=True)  # 顺时针旋转90度
    image = image.transpose(Image.FLIP_LEFT_RIGHT)  # #according to obj to flip

    return image, lm


# calculating least square problem
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1;

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def process_img(img, lm, t, s, target_size=224.):
    w0, h0 = img.size
    w = (w0 / s * 102).astype(np.int32)
    h = (h0 / s * 102).astype(np.int32)
    img = img.resize((w, h), resample=Image.BICUBIC)

    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * 102 / s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * 102 / s)).astype(np.int32)
    below = up + target_size

    img = img.crop((left, up, right, below))
    img = np.array(img)
    img = img[:, :, ::-1]  # RGBtoBGR
    img = np.expand_dims(img, 0)
    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) / s * 102
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img, lm, [left, right, up, below, w0 / w]


# resize and crop input images
def Preprocess(img, lm, lm3D):
    w0, h0 = img.size

    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm.transpose(), lm3D.transpose())

    # processing the image
    img_new, lm_new, posion = process_img(img, lm, t, s)
    lm_new = np.stack([lm_new[:, 0], 223 - lm_new[:, 1]], axis=1)
    trans_params = np.array([w0, h0, 102.0 / s, t[0], t[1]])

    return img_new, lm_new, trans_params, posion


def get_face_mask(img_norm, mask_model):
    img = torch.unsqueeze(img_norm, 0)
    img = img.cuda()
    out = mask_model(img)[0]
    parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
    face_mask = parsing.copy().astype(np.int32)
    # #check the face mask
    # plt.imshow(face_mask)
    # plt.show()

    parse_dilate = parsing.copy().astype(np.uint8)
    bg_index = (parse_dilate == 0)
    parse_face = parse_dilate.copy()
    parse_face[parse_face > 1] = 1
    parse_dilate[parse_dilate == 1] = 0
    kernel_2 = np.ones((10, 10), dtype=np.uint8)  # kernel size
    parse_dilate = cv2.dilate(parse_dilate, kernel_2, 1)
    parse_dilate = parse_dilate + parse_face
    parse_dilate[bg_index] = 0
    parse_dilate = parse_dilate.astype(np.float32)
    parse_dilate[parse_dilate == 0] = 0.25
    parse_dilate[parse_dilate == 1] = 0.50
    parse_dilate[parse_dilate > 1] = 0.99
    weight_map = parse_dilate.copy().astype(np.float32)
    # #check the weight map
    # plt.imshow(weight_map, cmap='gray')
    # plt.show()
    return face_mask, weight_map


def get_3d_gt_landmarks(txt_path):
    ld_single = []
    with open(txt_path, 'r') as fi:
        while (True):
            line = fi.readline().strip()
            if not line:
                break
            strs = line.split(' ')
            ld_single.append((float(strs[0]), float(strs[1]), float(strs[2])))
    fi.close()
    ld_106 = np.array(ld_single)
    ig_idx = np.load('lib/BFM/model_basis/BFM2009/index/ignore_index_5.npy')
    ld_101 = np.delete(ld_106, ig_idx, axis=0)
    kp_3d_gt = ld_101.copy().astype(np.float32)
    return kp_3d_gt


# ##################################################data process for shm#############################################
def Data_preprocess(list_name, mask_model):
    detector = MTCNN()

    # #split the img path and face_region
    img_path = list_name
    os.makedirs('dataset/pixel_face/crop_pic/', exist_ok=True)  # create dir
    crop_save_dir = 'dataset/pixel_face/crop_pic/' + list_name.split('/')[3] + list_name.split('/')[4] + \
                    list_name.split('/')[6] + '.png'

    # #normalized rgb to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    #  #detect the face and 5 landmarks, crop the img
    img_src, lm = load_img_and_lm(img_path, detector)  # detect
    lm3D = load_lm3d()  # get the 5 landmarks of 3d
    crop_img, lm_5n, trans_params, posion = Preprocess(img_src, lm, lm3D)

    # #save the crop image
    crop_img_raw = crop_img[0, :, :, :]
    cv2.imwrite(crop_save_dir, crop_img_raw)
    crop_img_raw = crop_img_raw[..., [2, 1, 0]]  # BGR to RGB
    # #check the cropped img
    # plt.imshow(crop_img_raw)
    # plt.show()

    # #normalize the input image
    img_norm = transform(crop_img_raw.copy())

    # #find the 68 landmarks of crop_img
    fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    pts_new = fa_3d.get_landmarks(np.array(crop_img_raw.copy()))
    if len(pts_new) < 1:
        assert "No face detected!"
    else:
        lm_68n_3d = np.array(pts_new[0]).astype(np.int32)
    kp_img_gt = lm_68n_3d[:, [0, 1]].astype(np.float32) / 224.0
    # #check the key points of img
    # plt.imshow(crop_img_raw)
    # plt.scatter(lm_68n_3d[:, 0], lm_68n_3d[:, 1], c='r', s=2)
    # plt.show()

    # #get face mask
    face_mask, weight_map = get_face_mask(img_norm, mask_model)

    # #expand dim for batch
    crop_img_raw = crop_img_raw / 255.0
    crop_img_raw = np.expand_dims(crop_img_raw, 0)
    img_norm = np.array(img_norm)
    img_norm = np.expand_dims(img_norm, 0)  # [1, 3, 244, 244] [-1, 1] RGB float32
    kp_img_gt = np.expand_dims(kp_img_gt, 0)  # [1, 68, 2] [0, 1] xy  float32
    face_mask = np.expand_dims(face_mask, 0)
    face_mask = np.expand_dims(face_mask, 0)  # [1, 1, 224, 224] [0-9] int32
    weight_map = np.expand_dims(weight_map, 0)
    weight_map = np.expand_dims(weight_map, 0)  # [1, 1, 224, 224] [0.25, 0.50, 0.99] float32

    if list_name.split('/')[6] == '1':
        # #get 101 3d landmarks
        kp_3d_path = os.path.join(list_name.split('/')[0], list_name.split('/')[1], list_name.split('/')[2],
                                  list_name.split('/')[3], list_name.split('/')[4], list_name.split('/')[5],
                                  'fusion', 'nicp_106_pts.txt')
        kp_3d_gt = get_3d_gt_landmarks(kp_3d_path)
        kp_3d_gt = np.expand_dims(kp_3d_gt, 0)  # [1, 101, 3]  xyz  float32
        return img_norm, crop_img_raw, kp_img_gt, face_mask, weight_map, kp_3d_gt
    else:
        return img_norm, crop_img_raw, kp_img_gt, face_mask, weight_map


# ############################################Dataset setting##########################################
class Dataset:
    def __init__(self, cfg):

        self.batch_size_train = cfg.batch_size_train
        self.batch_size_val = cfg.batch_size_val
        self.dataset_workers = cfg.num_works
        self.num_train_sample = cfg.num_train_sample
        self.dataset_choose = cfg.dataset
        self.cache = cfg.cache
        self.dist = cfg.dist
        self.mask_pred_path = cfg.mask_pred_path
        self.first_time = cfg.first_time
        self.color_aug = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, sharpness=0.5)

        if os.path.exists('dataset/' + self.dataset_choose + '/list_txt/shallow_path_val.npy'):
            self.train_file_list = np.load('dataset/' + self.dataset_choose + '/list_txt/train_file_list.npy').tolist()
            self.val_file_list = np.load('dataset/' + self.dataset_choose + '/list_txt/val_file_list.npy').tolist()
            self.shallow_path_train = np.load(
                'dataset/' + self.dataset_choose + '/list_txt/shallow_path_train.npy').tolist()
            self.shallow_path_val = np.load(
                'dataset/' + self.dataset_choose + '/list_txt/shallow_path_val.npy').tolist()
        else:
            self.train_file_list = glob.glob(r"dataset/pixel_face/train/*/*/*/*/t.jpg")
            self.val_file_list = glob.glob(r"dataset/pixel_face/val/*/*/*/*/t.jpg")
            self.shallow_path_train = glob.glob(r"dataset/pixel_face/train/*/*/*")
            self.shallow_path_val = glob.glob(r"dataset/pixel_face/val/*/*/*")
            os.makedirs('dataset/' + self.dataset_choose + '/list_txt/', exist_ok=True)  # create dir
            np.save('dataset/' + self.dataset_choose + '/list_txt/train_file_list.npy', np.array(self.train_file_list))
            np.save('dataset/' + self.dataset_choose + '/list_txt/val_file_list.npy', np.array(self.val_file_list))
            np.save('dataset/' + self.dataset_choose + '/list_txt/shallow_path_train.npy',
                    np.array(self.shallow_path_train))
            np.save('dataset/' + self.dataset_choose + '/list_txt/shallow_path_val.npy',
                    np.array(self.shallow_path_val))
        self.train_file_list.sort()
        self.val_file_list.sort()
        self.shallow_path_train.sort()
        self.shallow_path_val.sort()

        if self.first_time == True:
            self.mask_model = BiSeNet(10)
            self.mask_model.cuda()
            gpu = torch.cuda.current_device()
            map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
            self.mask_model.load_state_dict(torch.load(self.mask_pred_path, map_location=map_location))
            self.mask_model.eval()

            # create npy
            # os.makedirs('dataset/' + self.dataset_choose + '/npy/', exist_ok=True)  # create dir
            # create_npy(self.train_file_list, self.mask_model)
            # create_npy(self.val_file_list, self.mask_model)

            # # #delete shm
            # delete_shared_memory(self.train_file_list)
            # delete_shared_memory(self.val_file_list)

            # create shm
            if self.cache:
                create_shared_memory(self.train_file_list)
                create_shared_memory(self.val_file_list)

    def trainLoader(self):
        if self.num_train_sample is None:
            train_set = list(range(len(self.train_file_list) // 3))
        else:
            train_set = list(range(self.num_train_sample // 3))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size_train, collate_fn=self.trainMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
                                            drop_last=True, pin_memory=True,
                                            worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        val_set = list(range(len(self.val_file_list) // 3))
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None
        # self.val_sampler = None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size_val, collate_fn=self.valMerge,
                                          num_workers=self.dataset_workers,
                                          shuffle=(self.val_sampler is None), sampler=self.val_sampler, drop_last=False,
                                          pin_memory=True,
                                          worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)

    def color_augment(self, raw_ft, raw_lt, raw_rt, color_aug):
        raw_ft_img = np.floor(raw_ft[0, ...] * 255).astype(np.uint8)
        raw_ft_img = Image.fromarray(raw_ft_img)
        raw_lt_img = np.floor(raw_lt[0, ...] * 255).astype(np.uint8)
        raw_lt_img = Image.fromarray(raw_lt_img)
        raw_rt_img = np.floor(raw_rt[0, ...] * 255).astype(np.uint8)
        raw_rt_img = Image.fromarray(raw_rt_img)

        raw_ft_img_aug, raw_lt_img_aug, raw_rt_img_aug = color_aug(raw_ft_img, raw_lt_img, raw_rt_img)
        if np.random.random(1) < 0.3:
            raw_rt_img_aug = raw_rt_img_aug.filter(ImageFilter.GaussianBlur)
            raw_lt_img_aug = raw_lt_img_aug.filter(ImageFilter.GaussianBlur)
            raw_ft_img_aug = raw_ft_img_aug.filter(ImageFilter.GaussianBlur)
        raw_ft = np.array(raw_ft_img_aug).astype(np.float64)[np.newaxis, ...] / 255.0
        raw_lt = np.array(raw_lt_img_aug).astype(np.float64)[np.newaxis, ...] / 255.0
        raw_rt = np.array(raw_rt_img_aug).astype(np.float64)[np.newaxis, ...] / 255.0
        # if np.random.random(1) < 0.3:
        #     aug_para = np.random.uniform(0, 1.5, (1,3)) + 0.01
        #     raw_ft = raw_ft *aug_para
        #     raw_ft[raw_ft>1.0] =1.0
        #     raw_lt = raw_lt *aug_para
        #     raw_lt[raw_lt > 1.0] = 1.0
        #     raw_rt = raw_rt *aug_para
        #     raw_rt[raw_rt > 1.0] = 1.0
        input_ft = raw_ft.transpose(0, 3, 1, 2)*2 - 1.0
        input_lt = raw_lt.transpose(0, 3, 1, 2) * 2 - 1.0
        input_rt = raw_rt.transpose(0, 3, 1, 2) * 2 - 1.0

        return input_ft, input_lt, input_rt, raw_ft, raw_lt, raw_rt

    def trainMerge(self, id):
        # # left  view
        input_lt_batch = []
        raw_lt_batch = []
        kp_img_gt_lt_batch = []
        face_mask_lt_batch = []
        weight_map_lt_batch = []

        # # front  view
        input_ft_batch = []
        raw_ft_batch = []
        kp_img_gt_ft_batch = []
        face_mask_ft_batch = []
        weight_map_ft_batch = []

        # # rt  view
        input_rt_batch = []
        raw_rt_batch = []
        kp_img_gt_rt_batch = []
        face_mask_rt_batch = []
        weight_map_rt_batch = []

        kp_3d_gt_batch = []
        file_name_batch = []

        for i, idx in enumerate(id):
            file_name = self.shallow_path_train[idx]
            fn_0 = file_name.split('/')[3] + file_name.split('/')[4] + str(0)
            fn_1 = file_name.split('/')[3] + file_name.split('/')[4] + str(1)
            fn_2 = file_name.split('/')[3] + file_name.split('/')[4] + str(2)

            if self.cache:
                input_lt = SA.attach("shm://{}_img_norm".format(fn_0)).copy()
                raw_lt = SA.attach("shm://{}_img_raw".format(fn_0)).copy()
                kp_img_gt_lt = SA.attach("shm://{}_kp_img_gt".format(fn_0)).copy()
                face_mask_lt = SA.attach("shm://{}_face_mask".format(fn_0)).copy()
                weight_map_lt = SA.attach("shm://{}_weight_map".format(fn_0)).copy()

                input_ft = SA.attach("shm://{}_img_norm".format(fn_1)).copy()
                raw_ft = SA.attach("shm://{}_img_raw".format(fn_1)).copy()
                kp_img_gt_ft = SA.attach("shm://{}_kp_img_gt".format(fn_1)).copy()
                face_mask_ft = SA.attach("shm://{}_face_mask".format(fn_1)).copy()
                weight_map_ft = SA.attach("shm://{}_weight_map".format(fn_1)).copy()

                input_rt = SA.attach("shm://{}_img_norm".format(fn_2)).copy()
                raw_rt = SA.attach("shm://{}_img_raw".format(fn_2)).copy()
                kp_img_gt_rt = SA.attach("shm://{}_kp_img_gt".format(fn_2)).copy()
                face_mask_rt = SA.attach("shm://{}_face_mask".format(fn_2)).copy()
                weight_map_rt = SA.attach("shm://{}_weight_map".format(fn_2)).copy()

                kp_3d_gt = SA.attach("shm://{}_kp_3d_gt".format(fn_1)).copy()
                file_name_batch.append(fn_1)
            else:
                input_lt = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_0)).copy()
                raw_lt = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_0)).copy()
                kp_img_gt_lt = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_0)).copy()
                face_mask_lt = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_0)).copy()
                weight_map_lt = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_0)).copy()

                input_ft = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_1)).copy()
                raw_ft = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_1)).copy()
                kp_img_gt_ft = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_1)).copy()
                face_mask_ft = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_1)).copy()
                weight_map_ft = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_1)).copy()

                input_rt = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_2)).copy()
                raw_rt = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_2)).copy()
                kp_img_gt_rt = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_2)).copy()
                face_mask_rt = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_2)).copy()
                weight_map_rt = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_2)).copy()

                kp_3d_gt = np.load("dataset/pixel_face/npy/{}_lm_101n.npy".format(fn_1)).copy()
                file_name_batch.append(fn_1)
                pass
            input_ft, input_lt, input_rt, raw_ft, raw_lt, raw_rt = self.color_augment(raw_ft, raw_lt, raw_rt,
                                                                                      self.color_aug)
            # # for check input
            # plt.imshow((np.transpose(input_rt, [0, 2, 3, 1])[0, ...] + 1)/2.0)
            # plt.show()
            # plt.imshow(raw_rt[0, ...])
            # plt.scatter(kp_img_gt_rt[0, :, 0]*224, kp_img_gt_rt[0, :, 1]*224, c='b', s=2)
            # plt.show()
            # plt.imshow(face_mask_rt[0, 0, ...])
            # plt.show()
            # plt.imshow(weight_map_rt[0, 0, ...], cmap='gray')
            # plt.show()
            # #######change weight map from 0-1 to 0-255
            weight_map_lt[weight_map_lt == 0.25] = 32.0
            weight_map_lt[weight_map_lt == 0.50] = 128.0
            weight_map_rt[weight_map_rt == 0.99] = 254.0
            weight_map_ft[weight_map_ft == 0.25] = 32.0
            weight_map_ft[weight_map_ft == 0.50] = 128.0
            weight_map_ft[weight_map_ft == 0.99] = 254.0
            weight_map_rt[weight_map_rt == 0.25] = 32.0
            weight_map_rt[weight_map_rt == 0.50] = 128.0
            weight_map_rt[weight_map_rt == 0.99] = 254.0


            #  merge the scene to the batch
            input_lt_batch.append(torch.from_numpy(input_lt))
            raw_lt_batch.append(torch.from_numpy(raw_lt))
            kp_img_gt_lt_batch.append(torch.from_numpy(kp_img_gt_lt))
            face_mask_lt_batch.append(torch.from_numpy(face_mask_lt))
            weight_map_lt_batch.append(torch.from_numpy(weight_map_lt))

            input_ft_batch.append(torch.from_numpy(input_ft))
            raw_ft_batch.append(torch.from_numpy(raw_ft))
            kp_img_gt_ft_batch.append(torch.from_numpy(kp_img_gt_ft))
            face_mask_ft_batch.append(torch.from_numpy(face_mask_ft))
            weight_map_ft_batch.append(torch.from_numpy(weight_map_ft))

            input_rt_batch.append(torch.from_numpy(input_rt))
            raw_rt_batch.append(torch.from_numpy(raw_rt))
            kp_img_gt_rt_batch.append(torch.from_numpy(kp_img_gt_rt))
            face_mask_rt_batch.append(torch.from_numpy(face_mask_rt))
            weight_map_rt_batch.append(torch.from_numpy(weight_map_rt))

            kp_3d_gt_batch.append(torch.from_numpy(kp_3d_gt))
            pass

        #  # merge all the scenes in the batch
        input_lt_batch = torch.cat(input_lt_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_lt_batch = torch.cat(raw_lt_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_lt_batch = torch.cat(kp_img_gt_lt_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_lt_batch = torch.cat(face_mask_lt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_lt_batch = torch.cat(weight_map_lt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        input_ft_batch = torch.cat(input_ft_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_ft_batch = torch.cat(raw_ft_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_ft_batch = torch.cat(kp_img_gt_ft_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_ft_batch = torch.cat(face_mask_ft_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_ft_batch = torch.cat(weight_map_ft_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        input_rt_batch = torch.cat(input_rt_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_rt_batch = torch.cat(raw_rt_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_rt_batch = torch.cat(kp_img_gt_rt_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_rt_batch = torch.cat(face_mask_rt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_rt_batch = torch.cat(weight_map_rt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        kp_3d_gt_batch = torch.cat(kp_3d_gt_batch, 0).to(torch.float32)  # [B, 68, 3] float32 xyz

        return {'input_lt': input_lt_batch, 'input_ft': input_ft_batch, 'input_rt': input_rt_batch,
                'raw_lt': raw_lt_batch, 'raw_ft': raw_ft_batch, 'raw_rt': raw_rt_batch,
                'kp_img_gt_lt': kp_img_gt_lt_batch, 'kp_img_gt_ft': kp_img_gt_ft_batch,
                'kp_img_gt_rt': kp_img_gt_rt_batch, 'face_mask_lt': face_mask_lt_batch,
                'face_mask_ft': face_mask_ft_batch, 'face_mask_rt': face_mask_rt_batch,
                'weight_map_lt': weight_map_lt_batch, 'weight_map_ft': weight_map_ft_batch,
                'weight_map_rt': weight_map_rt_batch, 'kp_3d_gt': kp_3d_gt_batch, 'file_name': file_name_batch}

    def valMerge(self, id):
        # # left  view
        input_lt_batch = []
        raw_lt_batch = []
        kp_img_gt_lt_batch = []
        face_mask_lt_batch = []
        weight_map_lt_batch = []

        # # front  view
        input_ft_batch = []
        raw_ft_batch = []
        kp_img_gt_ft_batch = []
        face_mask_ft_batch = []
        weight_map_ft_batch = []

        # # rt  view
        input_rt_batch = []
        raw_rt_batch = []
        kp_img_gt_rt_batch = []
        face_mask_rt_batch = []
        weight_map_rt_batch = []

        kp_3d_gt_batch = []
        file_name_batch = []

        for i, idx in enumerate(id):
            file_name = self.shallow_path_val[idx]
            fn_0 = file_name.split('/')[3] + file_name.split('/')[4] + str(0)
            fn_1 = file_name.split('/')[3] + file_name.split('/')[4] + str(1)
            fn_2 = file_name.split('/')[3] + file_name.split('/')[4] + str(2)

            if self.cache:
                input_lt = SA.attach("shm://{}_img_norm".format(fn_0)).copy()
                raw_lt = SA.attach("shm://{}_img_raw".format(fn_0)).copy()
                kp_img_gt_lt = SA.attach("shm://{}_kp_img_gt".format(fn_0)).copy()
                face_mask_lt = SA.attach("shm://{}_face_mask".format(fn_0)).copy()
                weight_map_lt = SA.attach("shm://{}_weight_map".format(fn_0)).copy()

                input_ft = SA.attach("shm://{}_img_norm".format(fn_1)).copy()
                raw_ft = SA.attach("shm://{}_img_raw".format(fn_1)).copy()
                kp_img_gt_ft = SA.attach("shm://{}_kp_img_gt".format(fn_1)).copy()
                face_mask_ft = SA.attach("shm://{}_face_mask".format(fn_1)).copy()
                weight_map_ft = SA.attach("shm://{}_weight_map".format(fn_1)).copy()

                input_rt = SA.attach("shm://{}_img_norm".format(fn_2)).copy()
                raw_rt = SA.attach("shm://{}_img_raw".format(fn_2)).copy()
                kp_img_gt_rt = SA.attach("shm://{}_kp_img_gt".format(fn_2)).copy()
                face_mask_rt = SA.attach("shm://{}_face_mask".format(fn_2)).copy()
                weight_map_rt = SA.attach("shm://{}_weight_map".format(fn_2)).copy()

                kp_3d_gt = SA.attach("shm://{}_kp_3d_gt".format(fn_1)).copy()
                file_name_batch.append(fn_1)
            else:
                input_lt = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_0)).copy()
                raw_lt = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_0)).copy()
                kp_img_gt_lt = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_0)).copy()
                face_mask_lt = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_0)).copy()
                weight_map_lt = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_0)).copy()

                input_ft = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_1)).copy()
                raw_ft = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_1)).copy()
                kp_img_gt_ft = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_1)).copy()
                face_mask_ft = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_1)).copy()
                weight_map_ft = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_1)).copy()

                input_rt = np.load("dataset/pixel_face/npy/{}_img_norm.npy".format(fn_2)).copy()
                raw_rt = np.load("dataset/pixel_face/npy/{}_img_raw.npy".format(fn_2)).copy()
                kp_img_gt_rt = np.load("dataset/pixel_face/npy/{}_kp_img_gt.npy".format(fn_2)).copy()
                face_mask_rt = np.load("dataset/pixel_face/npy/{}_face_mask.npy".format(fn_2)).copy()
                weight_map_rt = np.load("dataset/pixel_face/npy/{}_weight_map.npy".format(fn_2)).copy()

                kp_3d_gt = np.load("dataset/pixel_face/npy/{}_lm_101n.npy".format(fn_1)).copy()
                file_name_batch.append(fn_1)
                pass
            # # for check input
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow((np.transpose(input_lt, [0, 2, 3, 1])[0, ...] + 1)/2.0)
            # plt.savefig('or_l.png', dpi=300)
            #
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(face_mask_lt[0, 0, ...])
            # plt.savefig('fm_l.png', dpi=300)
            #
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(weight_map_lt[0, 0, ...], cmap='gray')
            # plt.savefig('wm_l.png', dpi=300)

            # parse_dilate = face_mask_lt.copy().astype(np.uint8).squeeze(0)
            # bg_index = (parse_dilate == 0)
            # parse_face = parse_dilate.copy()
            # parse_face[parse_face > 1] = 1
            # parse_dilate[parse_dilate == 1] = 0
            # kernel_2 = np.ones((10, 10), dtype=np.uint8)  # kernel size
            # parse_dilate = cv2.dilate(parse_dilate, kernel_2, 1)
            # parse_dilate = parse_dilate + parse_face
            # parse_dilate[bg_index] = 0
            # parse_dilate = parse_dilate.astype(np.float32)
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(parse_dilate.squeeze(0))
            # plt.savefig('dl_l.png', dpi=300)
            #
            # parse_dilate = face_mask_ft.copy().astype(np.uint8).squeeze(0)
            # bg_index = (parse_dilate == 0)
            # parse_face = parse_dilate.copy()
            # parse_face[parse_face > 1] = 1
            # parse_dilate[parse_dilate == 1] = 0
            # kernel_2 = np.ones((10, 10), dtype=np.uint8)  # kernel size
            # parse_dilate = cv2.dilate(parse_dilate, kernel_2, 1)
            # parse_dilate = parse_dilate + parse_face
            # parse_dilate[bg_index] = 0
            # parse_dilate = parse_dilate.astype(np.float32)
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(parse_dilate.squeeze(0))
            # plt.savefig('dl_f.png', dpi=300)
            #
            # parse_dilate = face_mask_rt.copy().astype(np.uint8).squeeze(0)
            # bg_index = (parse_dilate == 0)
            # parse_face = parse_dilate.copy()
            # parse_face[parse_face > 1] = 1
            # parse_dilate[parse_dilate == 1] = 0
            # kernel_2 = np.ones((10, 10), dtype=np.uint8)  # kernel size
            # parse_dilate = cv2.dilate(parse_dilate, kernel_2, 1)
            # parse_dilate = parse_dilate + parse_face
            # parse_dilate[bg_index] = 0
            # parse_dilate = parse_dilate.astype(np.float32)
            # fig, ax = plt.subplots()
            # plt.axis('off')
            # fig.set_size_inches(224 / 100.0 / 3.0, 224 / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(parse_dilate.squeeze(0))
            # plt.savefig('dl_r.png', dpi=300)

            # #######change weight map from 0-1 to 0-255
            weight_map_lt[weight_map_lt == 0.25] = 32.0
            weight_map_lt[weight_map_lt == 0.50] = 128.0
            weight_map_rt[weight_map_rt == 0.99] = 254.0
            weight_map_ft[weight_map_ft == 0.25] = 32.0
            weight_map_ft[weight_map_ft == 0.50] = 128.0
            weight_map_ft[weight_map_ft == 0.99] = 254.0
            weight_map_rt[weight_map_rt == 0.25] = 32.0
            weight_map_rt[weight_map_rt == 0.50] = 128.0
            weight_map_rt[weight_map_rt == 0.99] = 254.0

            #  merge the scene to the batch
            input_lt_batch.append(torch.from_numpy(input_lt))
            raw_lt_batch.append(torch.from_numpy(raw_lt))
            kp_img_gt_lt_batch.append(torch.from_numpy(kp_img_gt_lt))
            face_mask_lt_batch.append(torch.from_numpy(face_mask_lt))
            weight_map_lt_batch.append(torch.from_numpy(weight_map_lt))

            input_ft_batch.append(torch.from_numpy(input_ft))
            raw_ft_batch.append(torch.from_numpy(raw_ft))
            kp_img_gt_ft_batch.append(torch.from_numpy(kp_img_gt_ft))
            face_mask_ft_batch.append(torch.from_numpy(face_mask_ft))
            weight_map_ft_batch.append(torch.from_numpy(weight_map_ft))

            input_rt_batch.append(torch.from_numpy(input_rt))
            raw_rt_batch.append(torch.from_numpy(raw_rt))
            kp_img_gt_rt_batch.append(torch.from_numpy(kp_img_gt_rt))
            face_mask_rt_batch.append(torch.from_numpy(face_mask_rt))
            weight_map_rt_batch.append(torch.from_numpy(weight_map_rt))

            kp_3d_gt_batch.append(torch.from_numpy(kp_3d_gt))
            pass

        #  # merge all the scenes in the batch
        input_lt_batch = torch.cat(input_lt_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_lt_batch = torch.cat(raw_lt_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_lt_batch = torch.cat(kp_img_gt_lt_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_lt_batch = torch.cat(face_mask_lt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_lt_batch = torch.cat(weight_map_lt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        input_ft_batch = torch.cat(input_ft_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_ft_batch = torch.cat(raw_ft_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_ft_batch = torch.cat(kp_img_gt_ft_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_ft_batch = torch.cat(face_mask_ft_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_ft_batch = torch.cat(weight_map_ft_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        input_rt_batch = torch.cat(input_rt_batch, 0).to(torch.float32)  # [B, 3, 224, 224] [-1, 1] float32 rgb
        raw_rt_batch = torch.cat(raw_rt_batch, 0).to(torch.float32)  # [B, 224, 224, 3] [0, 1] float32 rgb
        kp_img_gt_rt_batch = torch.cat(kp_img_gt_rt_batch, 0).to(torch.float32)  # [B, 68, 2] [0, 1] float32 xy
        face_mask_rt_batch = torch.cat(face_mask_rt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] [0-9] float32
        weight_map_rt_batch = torch.cat(weight_map_rt_batch, 0).to(torch.float32)  # [B, 1, 224, 224] float32

        kp_3d_gt_batch = torch.cat(kp_3d_gt_batch, 0).to(torch.float32)  # [B, 68, 3] float32 xyz

        return {'input_lt': input_lt_batch, 'input_ft': input_ft_batch, 'input_rt': input_rt_batch,
                'raw_lt': raw_lt_batch, 'raw_ft': raw_ft_batch, 'raw_rt': raw_rt_batch,
                'kp_img_gt_lt': kp_img_gt_lt_batch, 'kp_img_gt_ft': kp_img_gt_ft_batch,
                'kp_img_gt_rt': kp_img_gt_rt_batch, 'face_mask_lt': face_mask_lt_batch,
                'face_mask_ft': face_mask_ft_batch, 'face_mask_rt': face_mask_rt_batch,
                'weight_map_lt': weight_map_lt_batch, 'weight_map_ft': weight_map_ft_batch,
                'weight_map_rt': weight_map_rt_batch, 'kp_3d_gt': kp_3d_gt_batch, 'file_name': file_name_batch}
