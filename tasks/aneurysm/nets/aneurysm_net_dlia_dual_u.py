from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from nets.aneurysm_multi_view_3d import MultiView

class Dual_U(nn.Module):
    def __init__(self, segClasses=2, k=16, input_channel=1, psp=False, name=''):

        super(Dual_U, self).__init__()

        print('----------------- import dual unet ---------------')
        self.img_net = MultiView(segClasses=2, k=16, input_channel=1, psp=False, name=name)
        self.vessel_net = MultiView(segClasses=2, k=16, input_channel=1, psp=False, name=name)

    def forward(self, img, vessel):
        out1 = self.img_net(img)['y']
        out2 = self.vessel_net(vessel)['y']
        out = out1 + out2 * 0.1
        return {'y': out}