import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

base = {'1024': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512],
        '512': [],}
extras = {'1024': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
          '512': [],}
mbox = {'1024': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        '512': [],}

class SSD(nn.Module):
    def __init__(self, num_classes=8, size=1280, ssd_conf=None):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.cfg = ssd_conf
        self.priorbox = PriorBox(self.cfg)
        self.priors =  self.priorbox.forward().cuda()
        self.size = size
        vgg_base_layers = vgg(base[str(size)], 3)
        vgg_extra_layers = add_extras(extras[str(size)], 1024)
        base_, extras_, head_ = multibox(vgg_base_layers, vgg_extra_layers, mbox[str(size)], self.num_classes)
        self.vgg = nn.ModuleList(base_)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras_)
        self.loc = nn.ModuleList(head_[0])
        self.conf = nn.ModuleList(head_[1])

        self.softmax = nn.Softmax(dim=-1)
        #self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)# due to auto grad issue
        self.detect = Detect()

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        encoder = []

        for k in range(5):
            x = self.vgg[k](x)
        encoder.append(x)

        for k in range(5,16):
            x = self.vgg[k](x)
        encoder.append(x)

        for k in range(16,23):
            x = self.vgg[k](x)
        encoder.append(x)
        s = self.L2Norm(x)
        sources.append(s) #1028 160x160

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        encoder.append(x)
        sources.append(x) #1028 80x80

        idx = 1
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x) #1028 extra1 40x40, extra2 20x20, extra3 18x18, extra4 16x16,
                idx += 1
                if k == 1:
                    encoder.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.training:           
            mlocations = loc.view(loc.size(0), -1, 4)
            mconfidences = conf.view(conf.size(0), -1, self.num_classes)
            output = (mlocations, mconfidences, self.priors)
            return output, encoder
        else:
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                    self.priors.type(type(x.data)).cuda()  # default boxes 
                              )
            return output, encoder

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)
