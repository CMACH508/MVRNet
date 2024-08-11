from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from resunet import CBR, BasicBlock, DownSample, DANetHead
# from .resunet import CBR, BasicBlock, DownSample, DANetHead

class DAResUNet(nn.Module):

    def __init__(self, segClasses=2, k=16, input_channel=1, psp=False):

        super(DAResUNet, self).__init__()

        self.layer0 = CBR(input_channel, k, 7, 1)

        self.pool1 = DownSample(k, k, 'max')
        self.layer1 = nn.Sequential(
            BasicBlock(k, 2 * k),
            BasicBlock(2 * k, 2 * k)
        )

        self.pool2 = DownSample(2 * k, 2 * k, 'max')
        self.layer2 = nn.Sequential(
            BasicBlock(2 * k, 4 * k),
            BasicBlock(4 * k, 4 * k)
        )

        self.pool3 = DownSample(4 * k, 4 * k, 'max')
        self.layer3 = nn.Sequential(
            BasicBlock(4 * k, 8 * k, dilation=1),
            BasicBlock(8 * k, 8 * k, dilation=2),
            BasicBlock(8 * k, 8 * k, dilation=4)
        )

        # self.class3 = DANetHead(8 * k, 8 * k)
        self.class2 = nn.Sequential(
            BasicBlock(4 * k + 8 * k, 8 * k),
            CBR(8 * k, 4 * k, 1)
        )
        self.class1 = nn.Sequential(
            BasicBlock(2 * k + 4 * k, 4 * k),
            CBR(4 * k, 2 * k, 1)
        )
        self.class0 = nn.Sequential(
            BasicBlock(k + 2 * k, 2 * k),
            nn.Conv3d(2 * k, segClasses, kernel_size=1, bias=False)
        )
        self._init_weight()

        # refine
        self.up1 = nn.ConvTranspose3d(256, 256 // 2, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(128, 128 // 2, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(64, 2, 3, stride=2, padding=1, output_padding=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def refine(self, last_layer, current_layer, up_layer):
        # print('before refine {} {}'.format(last_layer.shape, current_layer.shape))
        up_sampled = up_layer(last_layer)
        # print('after refine', up_sampled.shape)
        # ones = torch.ones_like(up_sampled)
        # zeros = torch.zeros_like(up_sampled)
        mask = torch.where((0 < up_sampled) & (up_sampled < 1), 1, 0)
        refined = current_layer * mask + up_sampled * (1 - mask)
        return refined

    def forward(self, x):
        # print('input:', x.shape)  # [1, 1, 80, 80, 80]
        output0 = self.layer0(x)
        # print('output0:', output0.shape)  # [1, 32, 80, 80, 80]

        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0)
        # print('output1:', output1.shape)  # [1, 64, 40, 40, 40]

        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0)
        # print('output2:', output2.shape)  # [1, 128, 20, 20, 20]

        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0)
        # print('output3:', output3.shape)  # [1, 256, 10, 10, 10]

        # output = self.class3(output3)  # DAB
        output = output3
        # print('DAB output:', output.shape)  # [1, 256, 10, 10, 10]
        last_layer_1 = output
        # print('last_layer_1', last_layer_1.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class2(torch.cat([output2, output], 1))
        last_layer_2 = self.refine(last_layer_1, output, self.up1)
        # print('last_layer_2', last_layer_2.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class1(torch.cat([output1, output], 1))
        last_layer_3 = self.refine(last_layer_2, output, self.up2)
        # print('last_layer_3', last_layer_3.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class0(torch.cat([output0, output], 1))
        output = self.refine(last_layer_3, output, self.up3)
        # print('output', output.shape)

        return {'y': output}

if __name__ == '__main__':
    aa = torch.rand(size=(1, 1, 80, 80, 80))
    model = DAResUNet(k=32)
    out = model(aa)['y']
    print('out.shape=', out.shape)
