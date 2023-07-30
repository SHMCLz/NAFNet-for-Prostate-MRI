import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

from timm.models.layers import trunc_normal_


import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        # self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        # print('x4 .. ', x4.shape)

        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)

        # print('x5 .. ', x5.shape)
        # print(self.aspp5_bn)

        # x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()

        in_channel = 1
        # num_classes = 64
        if not beta:
            self.conv1 = self.conv(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(in_channel, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=2)
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise NotImplementedError
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_groups=num_groups, weight_std=weight_std, **kwargs)

    if pretrained:
        raise NotImplementedError

    # if pretrained:
    #     model_dict = model.state_dict()
    #     if num_groups and weight_std:
    #         pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
    #         overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    #         assert len(overlap_dict) == 312
    #     elif not num_groups and not weight_std:
    #         pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
    #         overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     else:
    #         raise ValueError('Currently only support BN or GN+WS')
    #     model_dict.update(overlap_dict)
    #     model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        raise NotImplementedError

    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



class RoIExtractor(nn.Module):

    def forward(self, feat, roi):
        #BxCxHxW ; Bx1xHxW
        # print('roi extractor .. ', feat.shape, roi.shape)
        x, y = torch.nonzero(roi[0, 0]).split(1, dim=1)

        return feat[:, :, x, y].permute(0, 2, 1, 3).squeeze(dim=3), x, y

        # return torch.stack([feat[:, :, 0, 0], feat[:, :, 0, 1], feat[:, :, 0, 2]], dim=1)

class ResnetMRIADCT2DWIRoImageHotMap(nn.Module):

    def __init__(self, out_channel=128):
        super().__init__()

        # global K
        # K = kernel_size

        self.roi_extractor = RoIExtractor()

        self.adc_feat_extractor = resnet50(pretrained=False, num_classes=out_channel)

        self.adc_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
            # nn.Conv2d(in_channels=out_channel, out_channel=2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        ])


        self.t2_feat_extractor = resnet50(pretrained=False, num_classes=out_channel)
        self.t2_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
        ])

        self.dwi_feat_extractor = resnet50(pretrained=False, num_classes=out_channel)
        self.dwi_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
        ])

        self.cnt = 0

        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def hotmap(self, bnc_feats, x, y, inp, roi, head, name='debug'):
        # bnc_feats: BxNxC ,
        # X, Y
        #inp: Nx1xHxW, roi: Nx1xHxW
        # print('inp , roi .. ', inp.shape, roi.shape)
        # print('X && Y .. ', x.shape, y.shape)
        B, N, C = bnc_feats.shape

        # if N < 1900:
        # print('too less pixels .. pass', N)
        # return
        # else:
        #     print('working on .. ', N)

        probs = head(bnc_feats.view(-1, C)) #Nx1
        # print('probs .. ', probs.shape)

        hotmap = torch.zeros_like(inp)
        hotmap[:, :, x, y] = probs



        # bnc_feats.view()
        # print('hotmap .. ', hotmap.shape, hotmap.sum(), hotmap.max(), hotmap.min())


        inp_img = (inp[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        inp_img = np.concatenate([inp_img[:, :, np.newaxis], inp_img[:, :, np.newaxis], inp_img[:, :, np.newaxis]], axis=2)
        roi_img = (roi[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        hotmap_gray = (hotmap[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        roi_x = (roi_img.astype(np.float32) / 255)[:, :, np.newaxis] #H, W, 1

        hotmap_gray = cv2.GaussianBlur(hotmap_gray, (5, 5), 0)
        # hotmap_gray = cv2.GaussianBlur(hotmap_gray, (31, 31), 0)
        k = 23
        kernel = np.ones((k, k ), np.float32) / (k * k)
        hotmap_gray= cv2.filter2D(hotmap_gray, -1, kernel)

        # hotmap_gray = ((hotmap_gray.astype(np.float32) / 255) ** 2) * 255
        # hotmap_gray = hotmap_gray.astype(np.uint8)



        # hotmap_gray = hotmap_gray * roi_x

        color_map_name = 'jet'
        colormap = plt.get_cmap(color_map_name)
        hotmap = (colormap(hotmap_gray) * 2 ** 8).astype(np.uint8)[:, :, :3]
        hotmap = cv2.cvtColor(hotmap, cv2.COLOR_RGB2BGR)

        # mixed = (inp_img.astype(np.float32) + hotmap.astype(np.float32)) / 2

        mixed = inp_img * (1 - roi_x) + hotmap * roi_x
        # print('mixed .. ', mixed.min(), mixed.max())
        mixed = mixed.round().astype(np.uint8)

        cv2.imwrite(f'/data/jupyter/mri/{name}_inp.png', inp_img)
        # cv2.imwrite(f'/data/jupyter/mri/{color_map_name}_{name}_roi.png', roi_img)
        # cv2.imwrite(f'/data/jupyter/mri/{color_map_name}_{name}_hotmap.png', hotmap)
        cv2.imwrite(f'/data/jupyter/mri/{name}_mixed.png', mixed)


    def forward(self, adc, adc_roi, t2, t2_roi, dwi, dwi_roi):
        assert adc.size(0) == 1 and t2.size(0) == 1
        # print('debug model input.. ', adc.shape, adc_roi.shape, t2.shape, t2_roi.shape)
        # print('adc .. ', adc.mean(), t2.mean(), dwi.mean(), adc_roi.mean(), t2_roi.mean(), dwi_roi.mean())

        adc_each_channel_roi = adc_roi.sum(dim=(0, 2, 3))

        # print(adc.shape, t2.shape)

        adc_feats = None

        self.cnt += 1

        cnt = 0

        for k in range(adc.size(1)):
            if adc_each_channel_roi[k] > 0:
                # print(adc[:, [k], :, :].shape, adc_roi[:, [k], :, :].shape)
                bchw_feats = self.adc_feat_extractor(adc[:, [k], :, :] * adc_roi[:, [k], :, :])

                # print('bchw_feats .. ', bchw_feats.shape, bchw_feats.mean())
                bnc_feats, x, y = self.roi_extractor(bchw_feats, adc_roi[:, [k], :, :])
                # print('bnc_feats.. ', bnc_feats.shape, bnc_feats.mean())
                self.hotmap(bnc_feats, x, y, adc[:, [k], :, :], adc_roi[:, [k], :, :], self.adc_head,
                            name=f'{self.cnt}-{k}-adc-resnet')

                B, N, C = bnc_feats.shape

                cnt += N

                cur_feats = bnc_feats.sum(dim=1)

                if adc_feats is None:
                    adc_feats = cur_feats
                else:
                    adc_feats += cur_feats

        adc_pred = self.adc_head(adc_feats / cnt)


        t2_each_channel_roi = t2_roi.sum(dim=(0, 2, 3))

        t2_feats = None

        cnt = 0
        for k in range(t2.size(1)):
            if t2_each_channel_roi[k] > 0:
                bchw_feats = self.t2_feat_extractor(t2[:, [k], :, :] * t2_roi[:, [k], :, :])
                bnc_feats, x, y = self.roi_extractor(bchw_feats, t2_roi[:, [k], :, :])

                self.hotmap(bnc_feats, x, y, t2[:, [k], :, :], t2_roi[:, [k], :, :], self.t2_head,
                            name=f'{self.cnt}-{k}-t2-resnet')

                B, N, C = bnc_feats.shape

                cnt += N

                cur_feats = bnc_feats.sum(dim=1)
                if t2_feats is None:
                    t2_feats = cur_feats
                else:
                    t2_feats += cur_feats

        t2_pred = self.t2_head(t2_feats / cnt)


        dwi_each_channel_roi = dwi_roi.sum(dim=(0, 2, 3))

        dwi_feats = None

        cnt = 0
        for k in range(dwi.size(1)):
            if dwi_each_channel_roi[k] > 0:
                bchw_feats = self.dwi_feat_extractor(dwi[:, [k], :, :])
                bnc_feats, x, y = self.roi_extractor(bchw_feats, dwi_roi[:, [k], :, :])

                self.hotmap(bnc_feats, x, y, dwi[:, [k], :, :], dwi_roi[:, [k], :, :], self.dwi_head,
                            name=f'{self.cnt}-{k}-dwi-resnet')
                B, N, C = bnc_feats.shape

                cnt += N

                cur_feats = bnc_feats.sum(dim=1)
                if dwi_feats is None:
                    dwi_feats = cur_feats
                else:
                    dwi_feats+= cur_feats

        # print('dwi cnt  ', cnt)

        dwi_pred = self.dwi_head(dwi_feats / cnt)

        # print('adc_pred .. ', adc_pred, t2_pred, dwi_pred)
        # print('.....', adc_each_channel_roi, t2_each_channel_roi, dwi_each_channel_roi, adc.size(1), t2.size(1), dwi.size(1))
        # exit(0)

        return 0.33333 * (adc_pred[:, 0] + t2_pred[:, 0] + dwi_pred[:, 0])


# class NAFNetLocal(Local_Base, NAFNet):
#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         NAFNet.__init__(self, *args, **kwargs)
#
#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    import resource
    def using(point=""):
        # print(f'using .. {point}')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        global Total, LastMem

        # if usage[2]/1024.0 - LastMem > 0.01:
        # print(point, usage[2]/1024.0)
        print(point, usage[2] / 1024.0)

        LastMem = usage[2] / 1024.0
        return usage[2] / 1024.0

    img_channel = 3
    width = 32
    
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width' , width)
    
    using('start . ')
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, 
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    using('network .. ')

    # for n, p in net.named_parameters()
    #     print(n, p.shape)


    inp = torch.randn((4, 3, 256, 256))

    out = net(inp)
    final_mem = using('end .. ')
    # out.sum().backward()

    # out.sum().backward()

    # using('backward .. ')

    # exit(0)

    inp_shape = (3, 512, 512)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

    print('total .. ', params * 8 + final_mem)



