# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .CrowdPoseKeypointsWithDae import CrowdPoseKeypointsWithDae as crowd_pose_kpt_dae
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator
from .target_generators import DaeGenerator


def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    # ! list of each scale output for heatmap
    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS + 2, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    # ! list of each scale output for ae
    joints_generator = [
        JointsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_JOINTS + 2,  # 默认是cfg.DATASET.NUM_JOINTS +2 表示两个锚点
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    dae_generator = [
        DaeGenerator(
            output_size, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    # ! train or trainval or test
    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        joints_generator,
        dae_generator,
        transforms
    )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    # ! None as default
    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
