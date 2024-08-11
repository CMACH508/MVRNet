from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F

from nets.resunet import CBR, BasicBlock, DownSample, DANetHead
# from .resunet import CBR, BasicBlock, DownSample, DANetHead

class DAResUNet(nn.Module):

    def __init__(self, segClasses=2, k=16, input_channel=1, psp=False):

        super(DAResUNet, self).__init__()

        print('----------------- import multi_view_add ---------------')

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
            BasicBlock(4 * k, 8 * k),
            BasicBlock(8 * k, 8 * k)
        )
        # self.layer3 = nn.Sequential(
            # BasicBlock(4 * k, 8 * k, dilation=1),
            # BasicBlock(8 * k, 8 * k, dilation=2),
            # BasicBlock(8 * k, 8 * k, dilation=4)
        # )

        # new add
        self.pool4 = DownSample(8 * k, 8 * k, 'max')
        self.layer4 = nn.Sequential(
            BasicBlock(8 * k, 16 * k, dilation=1),
            BasicBlock(16 * k, 16 * k, dilation=2),
            BasicBlock(16 * k, 16 * k, dilation=4)
        )

        self.dab = DANetHead(16 * k, 16 * k)

        # new add
        self.class3 = nn.Sequential(
            BasicBlock(8 * k + 16 * k, 16 * k),
            CBR(16 * k, 8 * k, 1)
        )

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


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # print(m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                # print(m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, out1_0, out2_0, out3_0, out4_0, draw=0):
        # print('input:', x.shape)  # [1, 1, 80, 80, 80]
        aa, bb = 1, 0.01
        output0 = self.layer0(x)
        # print('output0:', output0.shape)  # [1, 16, 80, 80, 80]

        output1_0 = self.pool1(output0)
        output1 = self.layer1(output1_0*aa + bb*out1_0)
        # print('output1:', output1.shape)  # [1, 32, 40, 40, 40]

        output2_0 = self.pool2(output1)
        output2 = self.layer2(output2_0*aa + bb*out2_0)
        # print('output2:', output2.shape)  # [1, 64, 20, 20, 20]

        output3_0 = self.pool3(output2)
        output3 = self.layer3(output3_0*aa + bb*out3_0)
        # print('output3:', output3.shape)  # [1, 128, 10, 10, 10]

        output4_0 = self.pool4(output3)
        output4 = self.layer4(output4_0*aa + bb*out4_0)
        # print('output4:', output4.shape)  # [1, 256, 5, 5, 5]

        output4 = self.dab(output4) # dab
        # output = self.dab(output3) # dab

        output = F.interpolate(output4, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class3(torch.cat([output3, output], 1))
        # output = self.class3(output3)  # DAB
        # print('up1 output:', output.shape)  # [1, 256, 10, 10, 10]
        # print('last_layer_1', last_layer_1.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class2(torch.cat([output2, output], 1))
        # print('last_layer_2', last_layer_2.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class1(torch.cat([output1, output], 1))
        # print('last_layer_3', last_layer_3.shape)

        output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=True)
        output = self.class0(torch.cat([output0, output], 1))
        # print('output', output.shape)

        return {'y': output}

if __name__ == '__main__':
    aa = torch.rand(size=(1, 1, 80, 80, 80))
    model = DAResUNet(k=16)
    out = model(aa)['y']
    print('out.shape=', out.shape)
