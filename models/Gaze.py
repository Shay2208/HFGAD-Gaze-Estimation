import torch
import torch.nn as nn

from .resnet import resnet18
from .GazeModules import HFGAD

dims = {'pvt_v2_b0':     [32, 64,  160, 256],
        'pvt_v2_b1':     [64, 128, 320, 512],
        'pvt_v2_b2':     [64, 128, 320, 512],
        'mobilenet_v2':   [64, 96, 160, 320],
        'resnet18':      [64, 128, 256, 512]}

class GazeModel(nn.Module):

    def __init__(self, backbone='resnet18'):
        super(GazeModel, self).__init__()
        #init encoder
        if backbone == 'resnet18':
            self.base_model = resnet18(pretrained=False)
            dim = dims['resnet18']
        elif backbone == 'mobilenet_v2':
            from .emcad import mobilenetv2
            self.base_model = mobilenetv2(pretrained=False)
            dim = dims['mobilenet_v2']
        elif backbone == 'pvt_v2_b0':
            from .PVT import pvt_v2_b0
            self.base_model = pvt_v2_b0(pretrained=False)
            dim = dims['pvt_v2_b0']
        elif backbone == 'pvt_v2_b1':
            from .PVT import pvt_v2_b1
            self.base_model = pvt_v2_b1(pretrained=False)
            dim = dims['pvt_v2_b1']
        elif backbone == 'pvt_v2_b2':
            from .PVT import pvt_v2_b2
            self.base_model = pvt_v2_b2(pretrained=False)
            dim = dims['pvt_v2_b2']
        elif backbone == 'pvt_v2_b2_li':
            from .PVT import pvt_v2_b2_li
            self.base_model = pvt_v2_b2_li(pretrained=False)
            dim = dims['pvt_v2_b2']
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        #init decoder
        self.decoder = HFGAD(dim)

    def forward(self, x_in):
        x = self.base_model(x_in)
        gaze = self.decoder(x)
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = nn.L1Loss()(gaze, label)
        return loss

import time
import cv2
import numpy as np
from torch.cuda.amp import autocast

if __name__ == '__main__':
    from thop import profile

    model = GazeModel().cuda()
    model.eval()
    input = torch.randn(10, 3, 224, 224).cuda()
    start = time.time()
    output = model(input)
    end = time.time()
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f'Time: {(end-start)*100:.2f} ms')
    print(f'flops: {flops/1e10:.2f} G, params: {params/1e6:.3f} M')
