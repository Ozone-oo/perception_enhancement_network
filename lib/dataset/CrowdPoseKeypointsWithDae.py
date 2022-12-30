# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by ManiaaJia
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import crowdposetools
from .CrowdPoseDataset import CrowdPoseDataset
from .target_generators import HeatmapGenerator


logger = logging.getLogger(__name__)


class CrowdPoseKeypointsWithDae(CrowdPoseDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 dae_generator,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT)

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 15, 'Number of joint with center for CrowdPose is 15'
        else:
            assert cfg.DATASET.NUM_JOINTS == 14, 'Number of joint for CrowdPose is 14'
        # ! =len(heatmap_generator)
        self.num_scales = self._init_check(heatmap_generator, dae_generator)
        self.num_joints = cfg.DATASET.NUM_JOINTS + 2 # ! 16
        self.with_center = cfg.DATASET.WITH_CENTER        
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints # ! 16        

        self.scale_aware_sigma = cfg.DATASET.SCALE_AWARE_SIGMA # ! False as default
        # ! only when scale_aware_sigma is used, they are used to compute sigma for each joints
        self.base_sigma = cfg.DATASET.BASE_SIGMA # ! 2 as default
        self.base_size = cfg.DATASET.BASE_SIZE # ! 256 as default
        self.int_sigma = cfg.DATASET.INT_SIGMA # ! False as default

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.dae_generator = dae_generator

    def __getitem__(self, idx):
        # ! anno with no transform
        img, anno = super().__getitem__(idx)

        # ! crowdpose mask is zero with (img_height & img_width), that is, no masks
        mask = self.get_mask(anno, idx)

        # ! is not crowd or have keypints
        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]
        
        # TODO(bowen): to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        # * now is people*16*3
        joints = self.get_joints(anno)
        
        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        target_list = list()
        dae_list = list()

        # ! 生成输出热图
        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id]) # ! 16*size*size
            dae_t = self.dae_generator[scale_id](joints_list[scale_id]) # ! type TBD

            target_list.append(target_t)
            dae_list.append(dae_t)
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)            
        
        return img, target_list, mask_list, dae_list

    def transform_crowd_to_ours(self, joints):
        #Crowd R leg:  11(ankle), 9(knee), 7(hip)
        #      L leg:  10(ankle), 8(knee), 6(hip)
        #      R arms: 5(wrist), 3(elbow), 1(shoulder)
        #      L arms: 4(wrist), 2(elbow), 0(shoulder)
        #      Head:   13 - upper neck, 12 - head top
        #      Torso:  15 - pelvis, 14 - thorax

        # Ours Head: 0 - head top, 1 - upper neck
        #      R arms: 2(shoulder), 3(elbow), 4(wrist)
        #      L arms: 5(shoulder), 6(elbow), 7(wrist)
        #      8 - thorax
        #      R leg:  9(hip), 10(knee), 11(ankle)
        #      L leg:  12(hip), 13(knee), 14(ankle)
        #      15 - pelvis

        crowd_to_ours = [12, 13, 1, 3, 5, 0 ,2 ,4, 14, 7, 9, 11, 6, 8, 10, 15]

        reordered_joints = np.zeros_like(joints)
        for ji in range(len(crowd_to_ours)):
            reordered_joints[:, ji, :] = \
                joints[:, crowd_to_ours[ji], :]

        return reordered_joints

    def get_joints(self, anno):
        num_people = len(anno)

        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :14, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            # ! 添加胸部(14)和髋部(15)
            if joints[i, 0, 2] and joints[i, 1, 2]:
                joints[i, 14, :2] = (joints[i, 0, :2] + joints[i, 1, :2]) / 2
                joints[i, 14, 2] = 1
            elif joints[i, 13, 2]:
                joints[i, 14, :] = joints[i, 13, :]
            
            if joints[i, 6, 2] and joints[i, 7, 2]:
                joints[i, 15, :2] = (joints[i, 6, :2] + joints[i, 7, :2]) / 2
                joints[i, 15, 2] = 1
            elif joints[i, 6, 2] or joints[i, 7, 2]:
                joints[i, 15, :] = joints[i, 6, :] if joints[i, 6, 2] else joints[i, 7, :]
            
            if self.with_center:
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                # ! if there any joint is visiable, compute the center_joint
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
            if self.scale_aware_sigma:
                # get person box
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.round(sigma + 0.5))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma
        return self.transform_crowd_to_ours(joints)

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        return m < 0.5

    def _init_check(self, heatmap_generator, dae_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(dae_generator, (list, tuple)), 'dae_generator should be a list or tuple'
        assert len(heatmap_generator) == len(dae_generator), \
            'heatmap_generator and dae_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(dae_generator)
            )
        return len(heatmap_generator)
