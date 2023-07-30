# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class MRIModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(MRIModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        self.parts = None

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None


        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.adc = data['adc'].to(self.device)
        self.adc_roi = data['adc_roi'].to(self.device)

        self.t2 = data['t2'].to(self.device)
        self.t2_roi = data['t2_roi'].to(self.device)

        self.dwi = data['dwi'].to(self.device)
        self.dwi_roi = data['dwi_roi'].to(self.device)
        # self.dwi = data['dwi'].to(self.device)
        # self.dwi_roi = data['dwi_roi'].to(self.device)

        self.label = data['label'].float().to(self.device)
        self.img_id = data['img_id'].int()


        # print('debug feed data .... ', self.adc.shape, self.adc_roi.shape, self.t2.shape, self.t2_roi.shape, self.dwi.shape, self.dwi_roi.shape, self.label.shape, self.label)


    def transpose(self, t, trans_idx):
        if trans_idx == 0:
            return t

        if trans_idx % 2 == 1:
            t = torch.flip(t, [3])

        if trans_idx // 2 % 2 == 1:
            t = torch.flip(t, [2])

        if trans_idx // 4 % 2 == 1:
            t = torch.rot90(t, 1, [2, 3])

        return t


        # if trans_idx // 24 % 2 == 1:
            # t = torch.rot90(t, 1, [2, 3])
            # t = torch.cat([t[:, 3:, :, :], t[:, :3, :, :]], dim=1)

        # idx = trans_idx // 4 % 6
        #
        # color_shuffle_idx = [[0, 1, 2, 3, 4, 5],
        # [0, 2, 1, 3, 5, 4],
        # [1, 0, 2, 4, 3, 5],
        # [1, 2, 0, 4, 5, 3],
        # [2, 0, 1, 5, 3, 4],
        # [2, 1, 0, 5, 4, 3]][idx]
        #
        # t = t[:, color_shuffle_idx, :, :]
        #
        # if trans_idx // 2 % 2 == 1:
        #     t = torch.flip(t, [3])
        #     t = torch.cat([t[:, 3:, :, :], t[:, :3, :, :]], dim=1) # swap left/right
        #
        # if trans_idx % 2 == 1:
        #     t = torch.flip(t, [2])
        #
        # return t

    # def transpose_inverse(self, t, trans_idx):
    #     if trans_idx == 0:
    #         return t
    #
    #     if trans_idx % 2 == 1:
    #         t = torch.flip(t, [2])
    #
    #     if trans_idx // 2 % 2 == 1:
    #         t = torch.flip(t, [3])
    #
    #     idx = trans_idx // 4 % 6
    #
    #     color_shuffle_idx = [[0, 1, 2, 3, 4, 5],
    #                          [0, 2, 1, 3, 5, 4],
    #                          [1, 0, 2, 4, 3, 5],
    #                          [2, 0, 1, 5, 3, 4], #BGR --> RGB
    #                          [1, 2, 0, 4, 5, 3], #BRG --> RGB
    #                          [2, 1, 0, 5, 4, 3]][idx]
    #
    #     t = t[:, color_shuffle_idx, :, :]
    #
    #     if trans_idx // 24 % 2 == 1:
    #         # t = torch.rot90(t, 3, [2, 3])
    #         t = torch.cat([t[:, 3:, :, :], t[:, :3, :, :]], dim=1)
    #
    #
    #     return t

    # def total_forward(self):
    #     b, c, h, w = self.gt.size()
    #     self.original_size = (b, c, h, w)
    #     assert b == 1
    #
    #     parts = []
    #     idxes = []
    #
    #     trans_num = self.opt['val'].get('trans_num', 1)
    #
    #     for trans_idx in range(trans_num):
    #         parts.append(self.transpose(self.lq[:, :, :, :], trans_idx))
    #         idxes.append({'i': 0, 'j': 0, 'ii': h, 'jj': w, 'trans_idx': trans_idx})
    #
    #     # for num in range(self.opt['val'].get('extra_num', 0)):
    #     #     import random
    #     #     trans_idx = random.randint(0, trans_num - 1)
    #     #     parts.append(self.transpose(self.lq[:, :, :, :], trans_idx))
    #     #     idxes.append({'i': 0, 'j': 0, 'ii': h, 'jj': w, 'trans_idx': trans_idx})
    #
    #     self.origin_lq = self.lq
    #     self.parts = parts
    #     self.idxes = idxes

    # def total_backward(self):
    #     preds = torch.zeros(self.original_size)
    #     b, c, h, w = self.original_size
    #     count_mt = torch.zeros((b, 1, h, w))
    #     # print(self.idxes, ',...')
    #     for cnt, each_idx in enumerate(self.idxes):
    #         i = each_idx['i']
    #         j = each_idx['j']
    #         ii = each_idx['ii']
    #         jj = each_idx['jj']
    #
    #         trans_idx = each_idx['trans_idx']
    #         preds[0, :, i:ii, j:jj] += self.transpose_inverse(self.outs[cnt].unsqueeze(0), trans_idx).squeeze(0)
    #         count_mt[0, 0, i:ii, j:jj] += 1.
    #
    #     self.output = preds / count_mt
    #     self.lq = self.origin_lq

    # def grids(self):
    #     b, c, h, w = self.gt.size()
    #     #     self.original_size = (b, c, h, w)
    #     #     assert b == 1
    #     # b, c, h, w = self.lq.size()
    #     self.original_size = (b, c, h, w)
    #
    #     # print('b c h w', b, c, h, w)
    #     assert b == 1
    #     if 'crop_size_h' in self.opt['val']:
    #         crop_size_h = self.opt['val']['crop_size_h']
    #     else:
    #         crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)
    #
    #     if 'crop_size_w' in self.opt['val']:
    #         crop_size_w = self.opt['val'].get('crop_size_w')
    #     else:
    #         crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)
    #
    #
    #     crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
    #     #adaptive step_i, step_j
    #     num_row = (h - 1) // crop_size_h + 1
    #     num_col = (w - 1) // crop_size_w + 1
    #
    #     import math
    #     step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
    #     step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)
    #
    #     # if step_j % self.scale != 0 and step_i % self.scale != 0:
    #     #     print('step are not 4k .... !!', step_i, step_j, crop_size_h, crop_size_w, num_row, num_col, h, w)
    #     # else:
    #     #     print('...... ', step_i, step_j)
    #
    #     while step_i % self.scale != 0:
    #         step_i -= 1
    #
    #     while step_j % self.scale != 0:
    #         step_j -= 1
    #
    #     scale = self.scale
    #
    #     # print('step_i, stepj', step_i, step_j)
    #     # exit(0)
    #
    #
    #     parts = []
    #     idxes = []
    #
    #     # cnt_idx = 0
    #
    #     i = 0  # 0~h-1
    #     last_i = False
    #     while i < h and not last_i:
    #         j = 0
    #         if i + crop_size_h >= h:
    #             i = h - crop_size_h
    #             last_i = True
    #
    #
    #         last_j = False
    #         while j < w and not last_j:
    #             # print('i, j .. ', i, j , h, w, crop_size_h, crop_size_w)
    #             if j + crop_size_w >= w:
    #                 j = w - crop_size_w
    #                 last_j = True
    #             # from i, j to i+crop_szie, j + crop_size
    #             # print(' trans 8')
    #             # print(f'{h} {w} , {i}, {j}, {i+crop_size_h}, {j+crop_size_w}, {crop_size_h}, {crop_size_w}')
    #             for trans_idx in range(self.opt['val'].get('trans_num', 1)):
    #                 parts.append(self.transpose(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale], trans_idx))
    #                 idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
    #                 # print('regular', 'parts .. last .. ', parts[-1].shape, 'i & j', i, j, 'trans idx', trans_idx, 'idxes lat',
    #                 #       idxes[-1], 'crop size ', crop_size_h, crop_size_w, 'gt size .. ', self.gt.size())
    #                 # print(parts[-1].shape)
    #                 # cnt_idx += 1
    #             j = j + step_j
    #         i = i + step_i
    #     if self.opt['val'].get('random_crop_num', 0) > 0:
    #         # print('... random crop num .. ', self.opt['val'].get('random_crop_num'))
    #         for _ in range(self.opt['val'].get('random_crop_num')):
    #             import random
    #             i = random.randint(0, h-crop_size_h)
    #             j = random.randint(0, w-crop_size_w)
    #
    #             i = i // self.scale * self.scale
    #             j = j // self.scale * self.scale
    #             # while i % self.scale != 0:
    #             #     i-= 1
    #
    #             # i, j = 220, 0
    #             trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
    #             parts.append(self.transpose(self.lq[:, :, i // scale: (i+ crop_size_h) // scale, j // scale: ( j + crop_size_w) // scale], trans_idx))
    #             idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx, 'crop_border': True})
    #             # print('parts .. last .. ', parts[-1].shape, 'i & j', i, j, 'trans idx', trans_idx, 'idxes lat', idxes[-1], 'crop size ', crop_size_h, crop_size_w, 'gt size .. ', self.gt.size())
    #
    #
    #     self.origin_lq = self.lq
    #     # self.lq = torch.cat(parts, dim=0)
    #     self.parts = parts
    #     # print('parts .. ', len(parts), self.lq.size())
    #     self.idxes = idxes
    #
    # def grids_inverse(self):
    #     preds = torch.zeros(self.original_size)
    #     b, c, h, w = self.original_size
    #
    #     count_mt = torch.zeros((b, 1, h, w))
    #     if 'crop_size_h' in self.opt['val']:
    #         crop_size_h = self.opt['val']['crop_size_h']
    #     else:
    #         crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)
    #
    #     if 'crop_size_w' in self.opt['val']:
    #         crop_size_w = self.opt['val'].get('crop_size_w')
    #     else:
    #         crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)
    #
    #     crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
    #
    #     # print('grids inverse .. crop_size h w ', crop_size_h, crop_size_w)
    #
    #     for cnt, each_idx in enumerate(self.idxes):
    #         i = each_idx['i']
    #         j = each_idx['j']
    #         trans_idx = each_idx['trans_idx']
    #         border_h = 0
    #         border_w = 0
    #
    #         if 'crop_border' in each_idx and each_idx['crop_border'] == True:
    #             border_h = crop_size_h // 3
    #             border_w = crop_size_w // 3
    #
    #
    #         # _, _, hh, ww = self.output.size()
    #         _, hh, ww = self.outs[cnt].size()
    #         # print('border_h, border _ w', border_h, border_w, preds[0, :, i + border_h:i + crop_size_h - border_h, j + border_w :j + crop_size_w - border_w].shape, self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)[:, border_h:hh-border_h, border_w:ww-border_w].shape )
    #         preds[0, :, i + border_h:i + crop_size_h - border_h, j + border_w :j + crop_size_w - border_w] += self.transpose_inverse(self.outs[cnt].unsqueeze(0), trans_idx).squeeze(0)[:, border_h:hh-border_h, border_w:ww-border_w]
    #         count_mt[0, 0, i + border_h:i + crop_size_h - border_h, j + border_w:j + crop_size_w - border_w] += 1.
    #
    #     self.output = (preds / count_mt).to(self.device)
    #     self.lq = self.origin_lq

    # def partial_mixup(self, input, gamma, indices):
    #     perm_input = input[indices]
    #     return input.mul(gamma).add(perm_input, alpha=1 - gamma)

    # def mixup_aug(self):
    #     # self.lq
    #     # self.gt
    #     import numpy as np
    #     # alpha = np.random.rand()
    #     alpha = 1.2 #from restormer
    #     lam = np.random.beta(alpha, alpha)
    #
    #     indices = torch.randperm(self.lq.size(0), device=self.lq.device, dtype=torch.long)
    #     self.lq = self.partial_mixup(self.lq, lam, indices)
    #     self.gt = self.partial_mixup(self.gt, lam, indices)

    def optimize_parameters(self, current_iter, tb_logger):
        update_freq = self.opt['train'].get('update_freq', 1)
        # self.optimizer_g.zero_grad()

        # if self.opt['train'].get('mixup', False):
        #     self.mixup_aug()



        preds = self.net_g(self.adc, self.adc_roi, self.t2, self.t2_roi, self.dwi, self.dwi_roi)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.label)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        # #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()

        if (current_iter + 1) % update_freq == 0:
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

            self.optimizer_g.step()
            self.optimizer_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        assert self.adc.size(0) == 1

        self.net_g.eval()
        with torch.no_grad():
            trans_num = self.opt['val'].get('trans_num', 1)

            pred = None

            # preds_xx = []

            for t in range(trans_num):
                cur_pred = self.net_g(self.transpose(self.adc, t), self.transpose(self.adc_roi, t), self.transpose(self.t2, t), self.transpose(self.t2_roi, t), self.transpose(self.dwi, t), self.transpose(self.dwi_roi, t))

                # preds_xx.append(float(cur_pred))

                # self.cam(input())



                if pred is None:
                    pred = cur_pred
                else:
                    pred += cur_pred
                # print('cur_pred .. ', cur_pred)

            pred = pred / trans_num
            # print(preds_xx)
            # pred = self.net_g(self.adc, self.adc_roi, self.t2, self.t2_roi, self.dwi, self.dwi_roi)
            # print('pred ... ', pred, trans_num)
            self.output = pred

        self.net_g.train()

    # def test(self):
    #     self.net_g.eval()
    #     with torch.no_grad():
    #         n = len(self.parts)
    #         outs = []
    #         m = self.opt['val'].get('max_minibatch', n)
    #         i = 0
    #         while i < n:
    #             j = i + m
    #             if j >= n:
    #                 j = n
    #             inp = torch.cat(self.parts[i:j], dim=0)
    #             # if self.opt['val'].get('expand_input'):
    #             #     b, _, h, w = inp.size()
    #             #     w_expand = w // 2
    #             #     h_expand = h // 2
    #             #     inp = F.pad(inp, (w_expand, w_expand, h_expand, h_expand), mode='reflect')
    #             pred = self.net_g(inp)
    #             if isinstance(pred, list):
    #                 pred = pred[-1]
    #             # if self.opt['val'].get('expand_input'):
    #             #     b, _, h, w = pred.size()
    #             #     pred = pred[:, :, h_expand*self.scale:-self.scale*h_expand, w_expand *self.scale:-self.scale*w_expand]
    #             for b_idx in range(pred.size(0)):
    #                 outs.append(pred[b_idx].detach().cpu())
    #             i = j
    #
    #         self.outs = outs
    #     self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        import os
        # self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image, self.opt['rank'])
        rank, world_size = get_dist_info()
        # print(' ... rankk .. ', rank)

        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        a_metric = 0.

        if rank == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    # print('cnt ... ', cnt)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt
                a_metric = metrics_dict[key]




            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)

        # tensor = self.probs

        def all_gather(data):
            import torch
            import torch.distributed as dist
            # import torch.distributed.get_world_size as get_world_size
            from torch.distributed import get_world_size
            import pickle

            """
            Run all_gather on arbitrary picklable data (not necessarily tensors)
            Args:
                data: any picklable object
            Returns:
                list[data]: list of data gathered from each rank
            """
            world_size = get_world_size()
            if world_size == 1:
                return [data]

            # serialized to a Tensor
            buffer = pickle.dumps(data)
            storage = torch.ByteStorage.from_buffer(buffer)
            tensor = torch.ByteTensor(storage).to("cuda")

            # obtain Tensor size of each rank
            local_size = torch.LongTensor([tensor.numel()]).to("cuda")
            size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            size_list = [int(size.item()) for size in size_list]
            max_size = max(size_list)

            # receiving Tensor from all ranks
            # we pad the tensor because torch all_gather does not support
            # gathering tensors of different shapes
            tensor_list = []
            for _ in size_list:
                tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
            if local_size != max_size:
                padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
                tensor = torch.cat((tensor, padding), dim=0)
            dist.all_gather(tensor_list, tensor)

            data_list = []
            for size, tensor in zip(size_list, tensor_list):
                buffer = tensor.cpu().numpy().tobytes()[:size]
                data_list.append(pickle.loads(buffer))

            return data_list

        probs = all_gather(self.probs)

        cls_labels = all_gather(self.cls_labels)


        auc = 0.


        if rank == 0:
            # print('probs .. ', probs.shape)
            probs = [x.detach().cpu() for x in probs]
            # for x in probs:
            #     print('prob .. .. ', x.shape)

            cls_labels = [x.detach().cpu() for x in cls_labels]

            probs = torch.cat(probs, dim=0)
            cls_labels = torch.cat(cls_labels, dim=0)

            # output = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
            # torch.distributed.gather(tensor=tensor, gather_list=output, dst=0)
            # output = torch.cat(output, dim=0)
            # print('output.shape .. ', output.shape)
            auc = self.calc_auc(probs, cls_labels)
            print('auc ... ', auc)
            # else:
            #     torch.distributed.gather(tensor=tensor, gather_list=[], dst=0)

        # torch.distributed.gather(self.probs, dst=0)

        return auc

    def calc_auc(self, probs, labels):
        # print('calc auc .. probs..', probs)
        # len(probs)

        probs = probs.numpy()
        labels = labels.numpy()
        # import numpy as np
        # labels = np.ones(len(probs)).astype(np.long)


        from sklearn import metrics

        # print('labels .. ', labels.shape, probs.shape, labels, probs)

        fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
        return metrics.auc(fpr, tpr)

        # length = 200
        # sum = 0.
        # for i in range(length):
        #     sum += 1 / length * (probs >= (i / length)).sum() / len(probs)
        # return sum


    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        probs = []
        cls_labels = []


        # from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
        # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # from pytorch_grad_cam.utils.image import show_cam_on_image

        # target_layers = [self.net_g.adc_feat_extractor]

        # self.cam = GradCAM(model=self.net_g, target_layers=target_layers, use_cuda=False)


        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            # if self.opt['val'].get('grids', False):
            #     self.grids()
            # else:
            #     self.total_forward()

            self.test()
            ### might be useful ...
            print('result', int(self.label), int(self.img_id), float(self.output))

            # if self.opt['val'].get('grids', False):
            #     self.grids_inverse()
            # else:
            #     self.total_backward()

            visuals = self.get_current_visuals()

            # print('.... val .. ', self.output.shape, self.output, self.label, self.label.shape)
            v = self.output
            # if float(self.label) == 0:
            #     v = 1 - v

            probs.append(v)
            cls_labels.append(self.label)


            #id, pred, pred_label, gt_label
            #log_dir:
            import os
            log_file = os.path.join(f'./results', self.opt['name'], f'{current_iter}.csv')

            dir_name = os.path.abspath(os.path.dirname(log_file))
            os.makedirs(dir_name, exist_ok=True)

            # print(self.img_id, v, v.round(), self.label, flush=True)

            file_object = open(log_file, 'a')
            file_object.write(f'{int(self.img_id)},{float(v)},{int(v.round())},{int(self.label)}\n')
            file_object.close()

            # sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            # if 'gt' in visuals:
            #     gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
            #     del self.gt

            # tentative for out of GPU memory
            del self.adc
            del self.t2
            del self.adc_roi
            del self.t2_roi
            del self.dwi
            del self.dwi_roi
            del self.output
            torch.cuda.empty_cache()

            # out_dict['result'] = self.output.detach().cpu()
            # if hasattr(self, 'label'):
            #     out_dict['label']

            # visuals['result']
            # visuals['label']

            # if save_img:
            #     if sr_img.shape[2] == 6:
            #         L_img = sr_img[:, :, :3]
            #         R_img = sr_img[:, :, 3:]
            #
            #         visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
            #
            #         imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
            #         imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
            #     else:
            #         if self.opt['is_train']:
            #
            #             save_img_path = osp.join(self.opt['path']['visualization'],
            #                                      img_name,
            #                                      f'{img_name}_{current_iter}.png')
            #
            #             save_gt_img_path = osp.join(self.opt['path']['visualization'],
            #                                      img_name,
            #                                      f'{img_name}_{current_iter}_gt.png')
            #         else:
            #             save_img_path = osp.join(
            #                 self.opt['path']['visualization'], dataset_name,
            #                 f'{img_name}.png')
            #             save_gt_img_path = osp.join(
            #                 self.opt['path']['visualization'], dataset_name,
            #                 f'{img_name}_gt.png')
            #
            #         imwrite(sr_img, save_img_path)
            #         imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                # if use_image:
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['label'], **opt_)


                # else:
                #     for name, opt_ in opt_metric.items():
                #         metric_type = opt_.pop('type')
                #         self.metric_results[name] += getattr(
                #             metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            # if cnt == 300:
            #     break
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test ')
        if rank == 0:
            pbar.close()

        current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        # collected_probs = OrderedDict()
        # collected_probs['probs'] = torch.cat(probs, dim=0)

        self.probs = torch.cat(probs, dim=0)
        self.cls_labels = torch.cat(cls_labels, dim=0)
        # print('... probs .. ', self.probs.shape)

        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['adc'] = self.adc.detach().cpu()
        out_dict['t2'] = self.t2.detach().cpu()
        out_dict['dwi'] = self.dwi.detach().cpu()
        # out_dict['dwi'] = self.dwi.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'label'):
            out_dict['label'] = self.label.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
