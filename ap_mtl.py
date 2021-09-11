import torch
import torch.nn as nn
import torch.nn.functional as F
from ssd import SSD

class Bottleneck(nn.Module):
    def __init__(self, in_planes):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes//4)
        self.conv2 = nn.Conv2d(in_planes//4, in_planes//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes//4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class ActivatedBatchNorm(nn.Module):
    def __init__(self, num_features, activation='relu', slope=0.01, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)

        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return x + chn_se + spa_se

class DecoderSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.ABN = ActivatedBatchNorm(middle_channels)
        self.SCSEBlock = SCSEBlock(middle_channels)
        self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, *args):
        x = torch.cat(args, 1)
        x = self.conv1(x)
        x = self.ABN(x)
        x = self.SCSEBlock(x)
        x = self.deconv(x)
        return x

class Segmentation_Decoder(nn.Module):

    def __init__(self, num_classes=21):
        super(Segmentation_Decoder, self).__init__()
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 3, 1, 1)
        self.lsm = nn.LogSoftmax(dim=1)

        self.bottle_neck1 = Bottleneck(256)
        self.bottle_neck2 = Bottleneck(512)
        self.bottle_neck3 = Bottleneck(1024)

        self.center = DecoderSCSE(512, 256, 256)
        self.decoder5 = DecoderSCSE(768, 512, 256)
        self.decoder4 = DecoderSCSE(512, 256, 128)
        self.decoder3 = DecoderSCSE(256, 128, 64)
        self.decoder2 = DecoderSCSE(128, 64, 64)
        self.decoder1 = DecoderSCSE(128, 64, 64)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, encoder):
        x, e1, e2, e3, e4 = encoder[0], self.bottle_neck1(encoder[1]), self.bottle_neck2(encoder[2]), self.bottle_neck3(encoder[3]), encoder[4]
        c = self.center(self.pool(e4))
        d4 = self.decoder5(c, e4)
        d3 = self.decoder4(d4, e3)
        d2 = self.decoder3(d3, e2)
        d1 = self.decoder2(d2, e1)
        x = F.upsample(x, d1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(d1, x)

        y = self.conv2(d1)
        y = self.tp_conv2(y)
        y_seg = self.lsm(y)
        return y_seg

class AP_MTL(nn.Module):
    def __init__(self, num_classes=8, size=1024, ssd_conf=None):
        super(AP_MTL, self).__init__()
        #detection
        self.detection = SSD(num_classes=num_classes, size=size, ssd_conf=ssd_conf)
        # segmentation decoder
        self.seg_decoder = Segmentation_Decoder(num_classes=num_classes)

    def forward(self, x):
        out_bbox, out_encoder = self.detection(x)
        out_seg = self.seg_decoder(out_encoder)
        return out_seg, out_bbox