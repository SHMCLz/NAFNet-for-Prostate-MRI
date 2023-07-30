import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

from timm.models.layers import trunc_normal_

K = 3

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        global K
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=K, padding=(K-1)//2, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=1, out_channel=64, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class RoIExtractor(nn.Module):

    def forward(self, feat, roi):
        #BxCxHxW ; Bx1xHxW
        # print('roi extractor .. ', feat.shape, roi.shape)
        x, y = torch.nonzero(roi[0, 0]).split(1, dim=1)

        return feat[:, :, x, y].permute(0, 2, 1, 3).squeeze(dim=3)

        # return torch.stack([feat[:, :, 0, 0], feat[:, :, 0, 1], feat[:, :, 0, 2]], dim=1)

class NAFNetMRIADCT2DWIRoImage(nn.Module):

    def __init__(self, img_channel=1, out_channel=128, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], kernel_size=3):
        super().__init__()

        global K
        K = kernel_size

        self.roi_extractor = RoIExtractor()

        self.adc_feat_extractor = NAFNet(img_channel=img_channel, out_channel=out_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)

        self.adc_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
            # nn.Conv2d(in_channels=out_channel, out_channel=2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        ])


        self.t2_feat_extractor = NAFNet(img_channel=img_channel, out_channel=out_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.t2_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
        ])

        self.dwi_feat_extractor = NAFNet(img_channel=img_channel, out_channel=out_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.dwi_head = nn.Sequential(*[
            nn.Linear(out_channel, 1),
            nn.Sigmoid()
        ])

        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, adc, adc_roi, t2, t2_roi, dwi, dwi_roi):
        assert adc.size(0) == 1 and t2.size(0) == 1
        # print('debug model input.. ', adc.shape, adc_roi.shape, t2.shape, t2_roi.shape)
        # print('adc .. ', adc.mean(), t2.mean(), dwi.mean(), adc_roi.mean(), t2_roi.mean(), dwi_roi.mean())

        adc_each_channel_roi = adc_roi.sum(dim=(0, 2, 3))

        # print(adc.shape, t2.shape)

        adc_feats = None

        cnt = 0

        for k in range(adc.size(1)):
            if adc_each_channel_roi[k] > 0:
                # print(adc[:, [k], :, :].shape, adc_roi[:, [k], :, :].shape)
                bchw_feats = self.adc_feat_extractor(adc[:, [k], :, :] * adc_roi[:, [k], :, :])

                # print('bchw_feats .. ', bchw_feats.shape, bchw_feats.mean())
                bnc_feats = self.roi_extractor(bchw_feats, adc_roi[:, [k], :, :])
                # print('bnc_feats.. ', bnc_feats.shape, bnc_feats.mean())

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
                bnc_feats = self.roi_extractor(bchw_feats, t2_roi[:, [k], :, :])

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
                bnc_feats = self.roi_extractor(bchw_feats, dwi_roi[:, [k], :, :])
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



