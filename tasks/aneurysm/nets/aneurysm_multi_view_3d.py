import torch
from torch import nn
import math, os


class ConvBR(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBR, self).__init__()
        self.cnn = nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=kSize,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.relu = nn.ReLU(True)
        # self.cbr = nn.Sequential(nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=kSize,
        #                                    stride=stride, padding=1, bias=False),
        #                          nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03),
        #                          nn.ReLU(True))

    def forward(self, x):
        out = self.cnn(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ConvBR_3d(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3, 3, 3), stride=1, dilation=1):
        super(ConvBR_3d, self).__init__()

        if not isinstance(kSize, tuple):
            kSize = (kSize, kSize, kSize)

        padding = (int((kSize[0] - 1) / 2) * dilation, int((kSize[1] - 1) / 2) * dilation, int((kSize[2] - 1) / 2) * dilation)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)

        return output


class CBRCB(nn.Module):
    def __init__(self, nIn, nOut, kSize=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(CBRCB, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=kSize,
                                           stride=stride, padding=padding, bias=False),
                                 nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03),
                                 nn.ReLU(True))
        self.cb = nn.Sequential(nn.Conv2d(in_channels=nOut, out_channels=nOut, kernel_size=kSize,
                                          stride=stride, bias=False, padding=padding),
                                nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03))
        self.act = nn.ReLU(True)

        self.downsample = None
        if nIn != nOut or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
            )

    def forward(self, x):
        out = self.cbr(x)
        out = self.cb(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        out = self.act(out)
        return out


class DownSample3D(nn.Module):
    def __init__(self, pool='max'):
        super(DownSample3D, self).__init__()
        if pool == 'max':
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool = pool

    def forward(self, input):
        output = self.pool(input)
        return output


class MultiView(torch.nn.Module):
    def __init__(self, segClasses=2, k=16, input_channel=1, psp=False, name=''):
        super(MultiView, self).__init__()
        type = name[-3:]
        if type == 'add':
            if 'refine' in name:
                from tasks.aneurysm.nets.aneurysm_net_dlia_refine_v2_mult_add import DAResUNet
                print('-------------------------- refine_multi_view_3d_add -----------------')
            else:
                from tasks.aneurysm.nets.aneurysm_net_dlia_mult_add import DAResUNet
                print('-------------------------- multi_view_3d_add -----------------')
        # elif type == 'cat':
        #     from tasks.aneurysm.nets.aneurysm_net_dlia_refine_v2_mult_cat import DAResUNet
        #     print('-------------------------- refine_multi_view_3d_cat -----------------')
        else:
            print('----------- model type error -----------')
        self.unet = DAResUNet(k=16, input_channel=input_channel)
        self.out_channels = [16, 32, 64, 128, 256]
        self.size = [80, 40, 20, 10, 5]
        self.cnn0_2d = ConvBR(input_channel, 16, 7, 1, 3)
        self.cnn0_3d = ConvBR_3d(16, 16)
        # self.cnn0_2d = CBRCB(1, 16)
        self.cnn1_2d = nn.Sequential(CBRCB(16, 32), CBRCB(32, 32))
        self.cnn1_3d = ConvBR_3d(32, 32)
        self.cnn2_2d = nn.Sequential(CBRCB(32, 64), CBRCB(64, 64))
        self.cnn2_3d = ConvBR_3d(64, 64)
        self.cnn3_2d = nn.Sequential(CBRCB(64, 128), CBRCB(128, 128))
        self.cnn3_3d = ConvBR_3d(128, 128)
        # self.cnn4_2d = nn.Sequential(CBRCB(128, 256), CBRCB(256, 256), CBRCB(256, 256))
        self.cnn1_down = DownSample3D('max')
        self.cnn2_down = DownSample3D('max')
        self.cnn3_down = DownSample3D('max')
        self.cnn4_down = DownSample3D('max')
        # self.cnn5_down = DownSample3D('max')

        self._init_weight()
        # paths = ['/cmach-data/lipeiying/program/_Aneurysm_/DLIA/dlia_refine_v2-3/model_epoch199.pth.tar',
        #          '/dssg/home/acct-eetsk/eetsk/lipeiying/_Aneurysm_/DLIA_Global/model_epoch199.pth.tar']
        # for path in paths:
        #     if os.path.exists(path):
        #         ckpt = torch.load(path)
        #         self.unet.load_state_dict(ckpt['model'])
        # print('---- unet load successfully ----')
        #
        # print('frozen unet parameters')
        # for pp in self.unet.parameters():
        #     pp.requires_grad = False

    def _init_weight(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def three_view_layer(self, x, layer, cnn_2d, cnn_3d):
        bs = x.shape[0]
        view_sum = torch.zeros(size=(bs, self.out_channels[layer], self.size[layer], self.size[layer], self.size[layer]),
                                   requires_grad=True).cuda()
        # view_sum = 0
        # print(view_sum.shape)
        for view_ii in range(3):
            if view_ii == 0:
                view_x = x.permute(2, 0, 1, 3, 4)
            elif view_ii == 1:
                view_x = x.permute(3, 0, 1, 2, 4)
            elif view_ii == 2:
                view_x = x.permute(4, 0, 1, 2, 3)
            view_out = torch.zeros(size=(self.size[layer], bs, self.out_channels[layer], self.size[layer], self.size[layer]),
                                   requires_grad=True).cuda()
            # print(view_out.shape)
            # print('view_x:', view_x.requires_grad)
            # print('0000view_out:', view_out.requires_grad)
            # print('  view_x.shape={} view_out.shape={}'.format(view_x.shape, view_out.shape))
            for i in range(self.size[layer]):
                # print('  view_x[i].shape=', view_x[i].shape, view_x.requires_grad)
                view_out[i] = cnn_2d(view_x[i])
                # print('  view_out[i].shape={}'.format(view_out[i].shape), view_x.requires_grad)
            if view_ii == 0:
                view_out = view_out.permute(1, 2, 0, 3, 4)
            elif view_ii == 1:
                view_out = view_out.permute(1, 2, 3, 0, 4)
            elif view_ii == 2:
                view_out = view_out.permute(1, 2, 3, 4, 0)
            # print('  view {}: shape={}'.format(view_ii, view_out.shape))
            # view_out = cnn_3d[view_ii](view_out)
            view_sum += view_out
        # print('view_out:', view_out.requires_grad)
        # print('view_sum:', view_sum.requires_grad)
        out = view_sum / 3
        out = cnn_3d(out)
        # print('out:', out.requires_grad)
        return out

    def forward(self, x, draw=0):
        # print('------')
        out0 = self.three_view_layer(x, 0, self.cnn0_2d, self.cnn0_3d)
        # print('after 1--- ', out0.shape)

        out1_0 = self.cnn1_down(out0)
        out1 = self.three_view_layer(out1_0, 1, self.cnn1_2d, self.cnn1_3d)
        # print('after 2--- ', out1_0.shape, out1.shape)

        out2_0 = self.cnn2_down(out1)
        out2 = self.three_view_layer(out2_0, 2, self.cnn2_2d, self.cnn2_3d)
        # print('after 3--- ', out2_0.shape, out2.shape)

        out3_0 = self.cnn3_down(out2)
        out3 = self.three_view_layer(out3_0, 3, self.cnn3_2d, self.cnn3_3d)
        # print('after 4--- ', out3_0.shape, out3.shape)

        out4_0 = self.cnn4_down(out3)
        # out4 = self.three_view_layer(out4_0, 4, self.cnn4_2d)
        # print('after 5--- ', out4_0.shape)

        out = self.unet(x, out1_0, out2_0, out3_0, out4_0, draw)
        return out


if __name__ == '__main__':

    model = MultiView(name='refine_multi_view_add', input_channel=2).cuda()
    # out = model(aa)['y']
    # print('out.shape=', out.shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    myloss = torch.nn.CrossEntropyLoss()
    label = torch.ones(size=(1, 80, 80, 80)).long().cuda()
    model.train()
    for ite in range(1):
        aa = torch.rand(size=(1, 2, 80, 80, 80)).cuda()
        out = model(aa)['y']
        print(out.shape)
        # loss = myloss(out, label)
        # optimizer.zero_grad()
        # for name, param in model.named_parameters():
            # if 'cnn2_2d.1.cbr.0.' in name:
            # if 'cnn2_3d.conv.weight' in name:
            #     print('=======')
            #     print('-->name:', name)
            #     print('-->param:', param[0][0])
            #     # print('-->requires_grad:', param.requires_grad)
            #     # print('-->grad:', param.grad)
        # loss.backward()
        # optimizer.step()
        # print('--------------- after update ------------')
        # for name, param in model.named_parameters():
        #     if 'cnn2_2d.1.cbr.0.' in name:
        #     # if 'cnn2_3d.conv.weight' in name:
        #         print('=======')
        #         print('-->name:', name)
        #         print('-->param:', param[0][0])
        #         print('-->sum:', torch.sum(param))
                # print('-->requires_grad:', param.requires_grad)
                # print('-->grad:', param.grad)