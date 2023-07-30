# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import (paired_paths_from_folder,
#                                     paired_paths_from_lmdb,
#                                     paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np
import os
import cv2

import imageio

import nibabel as nib

def read_niifile(niifilepath):
    try:
        img = nib.load(niifilepath)
        img_fdata = img.get_fdata()
        return img_fdata
    except:
        print('load error ... ', niifilepath)
        img_fdata = np.ones((5, 5, 5))
    return img_fdata

error_datas = [
9476404,
11172824,
11262427,

11313792,
11228464,
10968618,
10847971,
]

class PairedMRIDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedMRIDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


        # self.dataroot = opt['img_list']

        id_and_labels = []

        # import refile
        # with refile.smart_open(opt['id_and_labels'], 'r') as f:
        with open(opt['id_and_labels'], 'r', encoding='UTF-8-sig') as f:
            for each_line in f:
                img_id, label1, label2, label3 = each_line.split(',')
                # lr_id, hr_id = each_line.split(';')
                img_id, label1, label2, label3 = img_id.strip(), label1.strip(), label2.strip(), label3.strip()
                # lr_id, hr_id = lr_id.strip(), hr_id.strip()
                # data_list.append([lr_id, hr_id])
                img_id = int(img_id)
                if img_id in error_datas:
                    print('pass error data .. ', img_id)
                    continue
                id_and_labels.append([img_id, label1, label2, label3])

        self.id_and_labels = id_and_labels

        self.root = opt['dataroot']
        self.label_idx = opt['label_idx'] if 'label_idx' in opt else 0
        self.label_idx = int(self.label_idx)
        # self.eda =  opt['eda'] if 'eda' in opt else False

        self.datas = [None for _ in range(len(id_and_labels))]
        self.resizes = opt['resizes'] if 'resizes' in opt else None
        # self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        # if 'filename_tmpl' in opt:
        #     self.filename_tmpl = opt['filename_tmpl']
        # else:
        #     self.filename_tmpl = '{}'
        #
        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt[
        #     'meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = paired_paths_from_folder(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.filename_tmpl)
        # self.flag = self.opt['flag'] if 'flag' in self.opt else 'color'

    def resize(self, imgs, s):
        results = []
        for img in imgs:
            h, w, c = img.shape
            img_after = cv2.resize(img, (s, s))
            if c == 1:
                img_after = img_after[:, :, np.newaxis]
            results.append(img_after)
        return results

    def __getitem__(self, index):

        img_id, label1, label2, label3 = self.id_and_labels[index]
        label = [label1, label2, label3][self.label_idx]
        label = int(label)
        # eda_label, other_label = int(eda_label), int(other_label)

        if self.datas[index] is not None:
            adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi = self.datas[index]
        else:
            adc_img = read_niifile(os.path.join(self.root, 'Origin', 'ADC', f'{img_id}_ADC_IMAGE.nii')).astype(np.float32)
            adc_roi = read_niifile(os.path.join(self.root, 'Mask', 'ADC', f'{img_id}_ADC.nii')).astype(np.float32)

            dwi_img = read_niifile(os.path.join(self.root, 'Origin', 'DWI', f'{img_id}_DWI_IMAGE.nii'))
            dwi_roi = read_niifile(os.path.join(self.root, 'Mask', 'DWI', f'{img_id}_DWI.nii'))

            t2_img = read_niifile(os.path.join(self.root, 'Origin', 'T2', f'{img_id}_T2_IMAGE.nii')).astype(np.float32)
            t2_roi = read_niifile(os.path.join(self.root, 'Mask', 'T2', f'{img_id}_T2.nii')).astype(np.float32)




        adc_img = adc_img / adc_img.max()
        t2_img = t2_img / t2_img.max()
        dwi_img = dwi_img / dwi_img.max()

        h1, w1, c1 = adc_img.shape
        h1_, w1_, c1_ = adc_roi.shape

        h2, w2, c2 = t2_img.shape
        h2_, w2_, c2_ = t2_roi.shape

        if h1 != h1_ or w1 != w1_ or h2 != h2_ or w2 != w2_:
            print(index, img_id, 'adc .. ', adc_img.shape, adc_roi.shape, t2_img.shape, t2_roi.shape)
            return self.__getitem__(int(self.__len__() * np.random.rand()))

        # if h1 * w1 > 512*512 or h2 * w2 > 512*512:
        #     return self.__getitem__(int(self.__len__() * np.random.rand()))


        assert len(adc_img.shape) == 3 and len(t2_img.shape) == 3
        if len(dwi_img.shape) > 3:
            print('dwi_img and roi ', dwi_img.shape, dwi_roi.shape, index, img_id, adc_img.shape, t2_img.shape)
            dwi_img = dwi_img[:, :, :, 0, 0] #hardcode ..
            # print(0)

        adc_idxes = []
        for idx in range(adc_roi.shape[2]):
            if adc_roi[:, :, idx].sum() > 0:
                adc_idxes.append(idx)
        adc_img = adc_img[:, :, adc_idxes]
        adc_roi = adc_roi[:, :, adc_idxes]

        t2_idxes = []
        for idx in range(t2_roi.shape[2]):
            if t2_roi[:, :, idx].sum() > 0:
                t2_idxes.append(idx)
        t2_img = t2_img[:, :, t2_idxes]
        t2_roi = t2_roi[:, :, t2_idxes]

        dwi_idxes = []
        for idx in range(dwi_roi.shape[2]):
            if dwi_roi[:, :, idx].sum() > 0:
                dwi_idxes.append(idx)
        dwi_img = dwi_img[:, :, dwi_idxes]
        dwi_roi = dwi_roi[:, :, dwi_idxes]




        self.datas[index] = adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi
        # print('datas index is not none', self.datas[index] is not None)

        if self.opt['phase'] == 'train':
            # adc_img, adc_roi, t2_img, t2_roi = augment(
            #     [adc_img, adc_roi, t2_img, t2_roi], self.opt['use_flip'], self.opt['use_rot'])
            # print('debug .. train dataset .. ', img_id, adc_img.shape, adc_roi.shape, dwi_img.shape, dwi_roi.shape, t2_img.shape, t2_roi.shape)
            adc_choosed_idx = int(np.random.rand() * adc_img.shape[2])
            t2_choosed_idx = int(np.random.rand() * t2_img.shape[2])
            dwi_choosed_idx = int(np.random.rand() * dwi_img.shape[2])

            # print('....data', adc_img.shape, adc_roi.shape, t2_img.shape, t2_roi.shape)
            adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi = augment([adc_img[:, :, [adc_choosed_idx]], adc_roi[:, :, [adc_choosed_idx]], t2_img[:, :, [t2_choosed_idx]], t2_roi[:, :, [t2_choosed_idx]], dwi_img[:, :, [dwi_choosed_idx]], dwi_roi[:, :, [dwi_choosed_idx]] ], self.opt['use_flip'], self.opt['use_rot'])

            if 'gamma' in self.opt:
                gamma = np.random.rand() * 0.4 + 0.8
                # if np.random.rand() < 0.5:
                #     gamma = -gamma
                adc_img, t2_img, dwi_img = adc_img ** gamma, t2_img ** gamma, dwi_img ** gamma

        # print('load image', adc_img.shape, adc_roi.shape, adc_img.dtype, adc_roi.dtype)

        if self.resizes is not None:
            s = np.random.choice(self.resizes)
            adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi = self.resize([adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi], s)

        # print('after resize image', adc_img.shape, adc_roi.shape, adc_img.dtype, adc_roi.dtype)

        adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi = img2tensor([adc_img, adc_roi, t2_img, t2_roi, dwi_img, dwi_roi], bgr2rgb=False, float32=True)



        # self.datas[index] = {
        #     'adc': adc_img,
        #     'adc_roi': adc_roi,
        #     'dwi': dwi_img,
        #     'dwi_roi': dwi_roi,
        #     't2': t2_img,
        #     't2_roi': t2_roi,
        #     'label': eda_label if self.eda else other_label
        # }

        # return adc_img, adc_roi, dwi_img, dwi_roi, t2_img, t2_roi
        return {
            'adc': adc_img,
            'adc_roi': adc_roi,
            'dwi': dwi_img,
            'dwi_roi': dwi_roi,
            't2': t2_img,
            't2_roi': t2_roi,
            'label': float(label),
            'img_id': img_id
        }

    def __len__(self):
        return len(self.id_and_labels)
        # return len(self.paths)

