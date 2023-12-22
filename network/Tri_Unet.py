# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/12/29  下午8:32
# File Name: Tri_Unet
# IDE: PyCharm

import torch
import torch.nn as nn


# #attention
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, con_input):
        return self.conv(con_input)


class Tri_UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Tri_UNet, self).__init__()
        self.conv1_0 = DoubleConv(in_ch, 64)
        self.pool1_0 = nn.MaxPool2d(2)
        self.conv2_0 = DoubleConv(64, 128)
        self.pool2_0 = nn.MaxPool2d(2)
        self.conv3_0 = DoubleConv(128, 256)
        self.pool3_0 = nn.MaxPool2d(2)
        self.conv4_0 = DoubleConv(256, 512)
        self.pool4_0 = nn.MaxPool2d(2)
        self.conv5_0 = DoubleConv(512, 1024)

        self.conv1_1 = DoubleConv(in_ch, 64)
        self.pool1_1 = nn.MaxPool2d(2)
        self.conv2_1 = DoubleConv(64, 128)
        self.pool2_1 = nn.MaxPool2d(2)
        self.conv3_1 = DoubleConv(128, 256)
        self.pool3_1 = nn.MaxPool2d(2)
        self.conv4_1 = DoubleConv(256, 512)
        self.pool4_1 = nn.MaxPool2d(2)
        self.conv5_1 = DoubleConv(512, 1024)

        self.conv1_2 = DoubleConv(in_ch, 64)
        self.pool1_2 = nn.MaxPool2d(2)
        self.conv2_2 = DoubleConv(64, 128)
        self.pool2_2 = nn.MaxPool2d(2)
        self.conv3_2 = DoubleConv(128, 256)
        self.pool3_2 = nn.MaxPool2d(2)
        self.conv4_2 = DoubleConv(256, 512)
        self.pool4_2 = nn.MaxPool2d(2)
        self.conv5_2 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(3072, 512, 2, stride=2)
        self.conv6 = DoubleConv(2048, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(1024, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(512, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(256, out_ch)

        # attention
        self.att1_0 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att2_0 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att3_0 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.att4_0 = Attention_block(F_g=512, F_l=512, F_int=256)

        self.att1_1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att2_1 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att3_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.att4_1 = Attention_block(F_g=512, F_l=512, F_int=256)

        self.att1_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.att2_2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att3_2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.att4_2 = Attention_block(F_g=512, F_l=512, F_int=256)

    def forward(self, x0, x1, x2):
        # left face
        c1_0 = self.conv1_0(x0)
        p1_0 = self.pool1_0(c1_0)
        c2_0 = self.conv2_0(p1_0)
        p2_0 = self.pool2_0(c2_0)
        c3_0 = self.conv3_0(p2_0)
        p3_0 = self.pool3_0(c3_0)
        c4_0 = self.conv4_0(p3_0)
        p4_0 = self.pool4_0(c4_0)
        c5_0 = self.conv5_0(p4_0)

        # front face
        c1_1 = self.conv1_1(x1)
        p1_1 = self.pool1_1(c1_1)
        c2_1 = self.conv2_1(p1_1)
        p2_1 = self.pool2_1(c2_1)
        c3_1 = self.conv3_1(p2_1)
        p3_1 = self.pool3_1(c3_1)
        c4_1 = self.conv4_1(p3_1)
        p4_1 = self.pool4_1(c4_1)
        c5_1 = self.conv5_1(p4_1)

        # right face
        c1_2 = self.conv1_2(x2)
        p1_2 = self.pool1_2(c1_2)
        c2_2 = self.conv2_2(p1_2)
        p2_2 = self.pool2_2(c2_2)
        c3_2 = self.conv3_2(p2_2)
        p3_2 = self.pool3_2(c3_2)
        c4_2 = self.conv4_2(p3_2)
        p4_2 = self.pool4_2(c4_2)
        c5_2 = self.conv5_2(p4_2)

        marge5 = torch.cat([c5_0, c5_1, c5_2], dim=1)
        up_6 = self.up6(marge5)
        c4_0 = self.att4_0(g=up_6, x=c4_0)
        c4_1 = self.att4_1(g=up_6, x=c4_1)
        c4_2 = self.att4_2(g=up_6, x=c4_2)
        merge6 = torch.cat([up_6, c4_0, c4_1, c4_2], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3_0 = self.att3_0(g=up_7, x=c3_0)
        c3_1 = self.att3_1(g=up_7, x=c3_1)
        c3_2 = self.att3_2(g=up_7, x=c3_2)
        merge7 = torch.cat([up_7, c3_0, c3_1, c3_2], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2_0 = self.att2_0(g=up_8, x=c2_0)
        c2_1 = self.att2_1(g=up_8, x=c2_1)
        c2_2 = self.att2_2(g=up_8, x=c2_2)
        merge8 = torch.cat([up_8, c2_0, c2_1, c2_2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1_0 = self.att1_0(g=up_9, x=c1_0)
        c1_1 = self.att1_1(g=up_9, x=c1_1)
        c1_2 = self.att1_2(g=up_9, x=c1_2)
        merge9 = torch.cat([up_9, c1_0, c1_1, c1_2], dim=1)
        # c9 = self.conv9(merge9)
        return merge9