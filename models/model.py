import torch
import torch.nn as nn
import time
import copy
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.transforms import transforms

from .resnettr import resnet18
from thop import profile

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activation = None

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 在目标层注册前向和后向钩子
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 计算目标类别的梯度
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        # 获取Grad-CAM热力图
        if len(self.gradients.shape) == 4:  # [batch, channels, height, width]
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        elif len(self.gradients.shape) == 3:  # [batch, channels, height]
            weights = self.gradients.mean(dim=(2,), keepdim=True)
        elif len(self.gradients.shape) == 2:  # [batch, channels]
            weights = self.gradients.mean(dim=(1,), keepdim=True)
        else:
            raise ValueError(f"Unexpected gradient shape: {self.gradients.shape}")

        # weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        grad_cam = torch.sum(weights * self.activation, dim=1).squeeze()

        # 应用ReLU
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.cpu().detach().numpy()

        # 归一化
        grad_cam = cv2.resize(grad_cam, (224, 224))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False)

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)

    def forward(self, x_in):
        x1, x2, x3, x4, conv = self.base_model(x_in)

        feature = conv

        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = nn.L1Loss()(gaze, label)
        return loss

import time
from torch.cuda.amp import autocast
if __name__ == '__main__':
    model = Model().cuda()
    model.eval()
    input = torch.randn(10, 3, 224, 224).cuda()
    start = time.time()
    output = model(input)
    end = time.time()
    flops, params = profile(model, inputs=(input,), verbose=False)
    print(f'Time: {(end-start)*100:.2f} ms')
    print(f'flops: {flops/1e10:.2f} G, params: {params/1e6:.3f} M')
