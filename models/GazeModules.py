import torch
import torch.nn as nn

#efficient local attention
class ELA(nn.Module):   
    def __init__(self, in_channels):
        super(ELA, self).__init__()
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels//4, bias=False)
        self.GN = nn.GroupNorm(16, in_channels)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b,c,h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b,c,w)
        x_h = self.con1(x_h)    # [b,c,h]
        x_w = self.con1(x_w)    # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)   # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)   # [b, c, 1, w]
        return x_h * x_w * input
    
#
class DPW(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DPW, self).__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, 3, 2, 1, groups=in_channel)
        self.pwconv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.att = ELA(in_channel)
        self.norm = nn.BatchNorm2d(in_channel)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.att(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv(x)
        return x
    
#selective feature fusion
class SFM(nn.Module):
    def __init__(self, in_channels=512, groups=4):
        super(SFM, self).__init__()
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels//groups, bias=False)
        self.GN = nn.GroupNorm(16, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_f1, x_f2):
        B, C, H, W = x_f1.size()
        x = x_f1 + x_f2

        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)

        x_h_1 = self.con1(x_h)    # [b,c,h]
        x_w_1 = self.con1(x_w)    # [b,c,w]
        
        a_h_1 = self.sigmoid(self.GN(x_h_1)).view(B, C, H, 1)   # [b, c, h, 1]
        a_w_1 = self.sigmoid(self.GN(x_w_1)).view(B, C, 1, W)   # [b, c, 1, w]

        a_h_2 = 1 - a_h_1
        a_w_2 = 1 - a_w_1

        x_fusion = (a_h_1 * a_w_1 * x_f1) + (a_h_2 * a_w_2 * x_f2)
        return x_fusion

#multi-scale-gate grouping convolution
class MSGC(nn.Module):
    def __init__(self, in_channels, groups=8):
        super(MSGC, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.gconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//groups)
        self.gconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels//groups)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = x
        x2 = self.gconv1(x)
        x3 = self.gconv2(x)

        x1_weight = self.conv(self.pool(x1))
        x2_weight = self.conv(self.pool(x2))
        x3_weight = self.conv(self.pool(x3))

        x_weight = torch.cat([x1_weight, x2_weight, x3_weight], dim=2)
        x_weight = self.softmax(self.sigmoid(x_weight))

        x1_weight, x2_weight, x3_weight = torch.split(x_weight, [1, 1, 1], dim=2)
        x_out = (x1 * x1_weight) + (x2 * x2_weight) + (x3 * x3_weight)
        return x_out
    
class MSGC2(nn.Module):
    def __init__(self, in_channels, groups=8):
        super(MSGC2, self).__init__()
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//4, bias=False)
        self.con2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//4, bias=False)
        self.con3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//4, bias=False)
        self.gconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels//groups)
        self.gconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels//groups)
        
        self.GN = nn.GroupNorm(16, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def caculate_weight1(self, x):
        B, C, H, W = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
        x_h = self.con1(x_h)    # [b,c,h]
        x_w = self.con1(x_w)    # [b,c,w]
        x_h = self.GN(x_h).view(B, C, H, 1)   # [b, c, h, 1]
        x_w = self.GN(x_w).view(B, C, 1, W)   # [b, c, 1, w]
        return x_h * x_w
    
    def caculate_weight2(self, x):
        B, C, H, W = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
        x_h = self.con2(x_h)    # [b,c,h]
        x_w = self.con2(x_w)    # [b,c,w]
        x_h = self.GN(x_h).view(B, C, H, 1)   # [b, c, h, 1]
        x_w = self.GN(x_w).view(B, C, 1, W)   # [b, c, 1, w]
        return x_h * x_w
    
    def caculate_weight3(self, x):
        B, C, H, W = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
        x_h = self.con3(x_h)    # [b,c,h]
        x_w = self.con3(x_w)    # [b,c,w]
        x_h = self.GN(x_h).view(B, C, H, 1)   # [b, c, h, 1]
        x_w = self.GN(x_w).view(B, C, 1, W)   # [b, c, 1, w]
        return x_h * x_w

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = x
        x2 = self.gconv1(x)
        x3 = self.gconv2(x)

        x1_weight = self.caculate_weight1(x1)
        x2_weight = self.caculate_weight2(x2)
        x3_weight = self.caculate_weight3(x3)
        x_weight = torch.cat([x1_weight, x2_weight, x3_weight], dim=1)
        x_weight = self.softmax(self.sigmoid(x_weight))

        x1_weight, x2_weight, x3_weight = torch.split(x_weight, x.size(1), dim=1)
        x_out = (x1 * x1_weight) + (x2 * x2_weight) + (x3 * x3_weight)
        return x_out


class EMGA(nn.Module):
    def __init__(self, in_channels, groups=8):
        super(EMGA, self).__init__()
        self.ela = ELA(in_channels//2)
        self.msgc = MSGC(in_channels//2, groups)
        self.conv = nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=0, groups=in_channels//groups)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.ela(x1)
        x1 = self.msgc(x1)
        x2 = self.msgc(x2)
        x = torch.cat([x1, x2], dim=1)
        return x
        
# Hierarchical Fine-Grained Attention Decoder (HFGAD)
# Paper: HFGAD: Hierarchical Fine-Grained Attention Decoder for Gaze Estimation
#        (Algorithms 2025, 18, 538)
# NOTE: this module was named `SRAD` during development. The state_dict keys are
# unchanged, so previously released `*.pt` checkpoints still load as-is.
class HFGAD(nn.Module):

    def __init__(self, dim):
        super(HFGAD, self).__init__()
        self.sfm = nn.ModuleList([SFM(dim[1]),
                                  SFM(dim[2]),
                                  SFM(dim[3])])
        self.dpw = nn.ModuleList([DPW(dim[0], dim[1]),
                                  DPW(dim[1], dim[2]),
                                  DPW(dim[2], dim[3])])
        # self.att1 = MSGC2(dim[0])
        # self.att2 = MSGC2(dim[1])
        # self.att3 = MSGC2(dim[2])
        # self.att4 = MSGC2(dim[3])
        self.att1 = nn.Sequential(ELA(dim[0]),
                                  MSGC(dim[0]))
        self.att2 = nn.Sequential(ELA(dim[1]),
                                  MSGC(dim[1]))
        self.att3 = nn.Sequential(ELA(dim[2]),
                                  MSGC(dim[2]))
        self.att4 = nn.Sequential(ELA(dim[3]),
                                  MSGC(dim[3]))
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(dim[3], 2)
    
    def forward(self, x_in):
        x1, x2, x3, x4 = x_in

        x1 = self.att1(x1)
        x1 = self.dpw[0](x1)                  # B, dim[0], 56, 56

        x2 = self.sfm[0](x1, x2)              # B, dim[1], 56, 56
        x2 = self.att2(x2)
        x2 = self.dpw[1](x2)                  # B, dim[1], 28, 28

        x3 = self.sfm[1](x2, x3)              # B, dim[2], 28, 28
        x3 = self.att3(x3)
        x3 = self.dpw[2](x3)                  # B, dim[2], 14, 14

        x4 = self.sfm[2](x3, x4)              # B, dim[3], 7 , 7
        x4 = self.att4(x4)

        f = self.pool(x4).flatten(1)
        gaze = self.fc(f)                     # (dim[3], 2)

        return gaze


# Backward-compatible alias (old checkpoints / scripts may still import `SRAD`).
SRAD = HFGAD


if __name__ == '__main__':
    from thop import profile

    decoder = HFGAD([64, 128, 256, 512]).cuda()
    decoder.eval()
    x1 = torch.randn(1, 64, 56, 56).cuda()
    x2 = torch.randn(1, 128, 28, 28).cuda()
    x3 = torch.randn(1, 256, 14, 14).cuda()
    x4 = torch.randn(1, 512, 7, 7).cuda()
    x_in = (x1, x2, x3, x4)
    input = torch.randn(1, 512, 7, 7)
    flops, params = profile(decoder, inputs=(x_in,), verbose=False)
    print('flops: {:.4f}G, params: {:.4f}M'.format(flops/1e9, params/1e6))
