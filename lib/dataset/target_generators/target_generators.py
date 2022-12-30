# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) # ! 钟形

    def __call__(self, joints):
        # ! 14*size*size
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        # ! p means every people: 14*3
        for p in joints:
            for idx, pt in enumerate(p):
                # ! any joint
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    # ! 去除不合理关键点
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d]) # ! 钟盖上去取max
        return hms


class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms


class JointsGenerator():
    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint # ! True as default

    def __call__(self, joints):
        # ! max_num_people*14*2, 计算位置的地方
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))
        output_res = self.output_res
        # ! i means iteration with people_num
        for i in range(len(joints)):
            tot = 0
            # ! idx from 0 to 13
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                # ! the joint is visible
                if pt[2] > 0 and x >= 0 and y >= 0 \
                   and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = \
                            (idx * output_res**2 + y * output_res + x, 1)
                    else:
                        visible_nodes[i][tot] = \
                            (y * output_res + x, 1)
                    tot += 1
        return visible_nodes


# ! conn_heatmaps, 前两张是上到下, 后两张是下到上
class DaeGenerator():
    def __init__(self, output_res, sigma=-1):
        self.output_res = output_res # ! 需要
        self.sigma = self.output_res/64 if sigma<0 else sigma
    
    # ! 传入DAE形式的关节列表(已经经过了相应的output_size坐标换算), 返回生成的热图
    def __call__(self, joints_list):
        output_res = self.output_res
        sigma = self.sigma
        num_people, num_joints, _ = joints_list.shape

        offset_maps = np.zeros((num_joints * 2, output_res, output_res), dtype=np.float32) # ! 32*out_size*out_size
        offset_weight_maps = np.zeros((num_joints * 2, output_res, output_res), dtype=np.float32)

        conn_maps = np.zeros((4, output_res, output_res), dtype=np.float32) # ! 4*out_size*out_size
        conn_weight_maps = np.zeros((4, output_res, output_res), dtype=np.float32)

        # ! data ready to compute offset_maps
        offset_joints_list = np.zeros((num_people, num_joints, 5), dtype = np.float32)
        offset_joints_list[:, :, :3] = joints_list
        for pi in range(num_people):
            # ! coco
            if joints_list[pi, 17, 2]==0 or \
                joints_list[pi, 17, 0]<0 or \
                joints_list[pi, 17, 1]<0 or \
                joints_list[pi, 17, 0]>=output_res or \
                joints_list[pi, 17, 1]>=output_res:
                pass
            else:
                offset_joints_list[pi, :11, 3:] = joints_list[pi, 17, :2]
            if joints_list[pi, 18, 2]==0 or \
                joints_list[pi, 18, 0]<0 or \
                joints_list[pi, 18, 1]<0 or \
                joints_list[pi, 18, 0]>=output_res  or \
                joints_list[pi, 18, 1]>=output_res:
                pass
            else:
                offset_joints_list[pi, 11:, 3:] = joints_list[pi, 18, :2]
            # ! crowdpose
            # if joints_list[pi, 8, 2]==0 or \
            #     joints_list[pi, 8, 0]<0 or \
            #     joints_list[pi, 8, 1]<0 or \
            #     joints_list[pi, 8, 0]>=output_res or \
            #     joints_list[pi, 8, 1]>=output_res:
            #     pass
            # else:
            #     offset_joints_list[pi, :9, 3:] = joints_list[pi, 8, :2]
            # if joints_list[pi, 15, 2]==0 or \
            #     joints_list[pi, 15, 0]<0 or \
            #     joints_list[pi, 15, 1]<0 or \
            #     joints_list[pi, 15, 0]>=output_res  or \
            #     joints_list[pi, 15, 1]>=output_res:
            #     pass
            # else:
            #     offset_joints_list[pi, 9:, 3:] = joints_list[pi, 15, :2]

        for ji in range(num_joints):
            data = offset_joints_list[:, ji, :]
            offset_maps[ji*2:(ji+1)*2, :], offset_weight_maps[ji*2:(ji+1)*2, :] = \
                self.gen_offset_map(data, output_res, sigma)

        # ! data ready to compute conn_maps, 0 means down(pelvis) is dae, 1 means top(thorax) is dae
        conn_joints_list = np.zeros((num_people, 2, 5), dtype = np.float32)
        conn_joints_list[:, 0, :3] = joints_list[:, 17, :]
        conn_joints_list[:, 1, :3] = joints_list[:, 18, :]
        for pi in range(num_people):
            # ! crowdpose
            # if joints_list[pi, 15, 2]==0 or \
            #     joints_list[pi, 15, 0]<0 or \
            #     joints_list[pi, 15, 1]<0 or \
            #     joints_list[pi, 15, 0]>=output_res  or \
            #     joints_list[pi, 15, 1]>=output_res:
            #     pass
            # else:
            #     conn_joints_list[pi, 0, 3:] = joints_list[pi, 15, :2]
            # if joints_list[pi, 8, 2]==0 or \
            #     joints_list[pi, 8, 0]<0 or \
            #     joints_list[pi, 8, 1]<0 or \
            #     joints_list[pi, 8, 0]>=output_res or \
            #     joints_list[pi, 8, 1]>=output_res:
            #     pass
            # else:
            #     conn_joints_list[pi, 1, 3:] = joints_list[pi, 8, :2]
            # ! coco
            if joints_list[pi, 18, 2]==0 or \
                joints_list[pi, 18, 0]<0 or \
                joints_list[pi, 18, 1]<0 or \
                joints_list[pi, 18, 0]>=output_res  or \
                joints_list[pi, 18, 1]>=output_res:
                pass
            else:
                conn_joints_list[pi, 0, 3:] = joints_list[pi, 18, :2]
            if joints_list[pi, 17, 2]==0 or \
                joints_list[pi, 17, 0]<0 or \
                joints_list[pi, 17, 1]<0 or \
                joints_list[pi, 17, 0]>=output_res or \
                joints_list[pi, 17, 1]>=output_res:
                pass
            else:
                conn_joints_list[pi, 1, 3:] = joints_list[pi, 17, :2]

        for ji in range(2):
            data = conn_joints_list[:, ji, :]
            conn_maps[ji*2:(ji+1)*2, :], conn_weight_maps[ji*2:(ji+1)*2, :] = \
                self.gen_offset_map(data, output_res, sigma)        

        return offset_maps, offset_weight_maps, conn_maps, conn_weight_maps

    def gen_offset_map(self, data, output_res, sigma):
        orientation_map = np.zeros((2, output_res, output_res))
        orientation_weight_map = np.zeros((2, output_res, output_res))
    
        # ! 对每个关节点和质心对
        for i in range(data.shape[0]):
            if data[i, 2] == 0 or data[i, 0]<0 or data[i, 1]<0 \
                or data[i, 0]>=output_res or data[i, 1]>=output_res \
                or (data[i, 3]==0 and data[i, 4]==0):
                continue
            x, y = int(data[i, 0]), int(data[i, 1]) # * joint
            objpos = int(data[i, 3]), int(data[i, 4]) # * anchor joint

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
            
            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)

            for i in range(cc, dd):
                for j in range(aa, bb):
                    # d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])                    
                    # exponent = d2 / 2.0 / sigma / sigma
                    # if exponent > 4.6052: # 超出一定范围则不赋值
                    #     continue
                    
                    # 否则赋值为相对偏移
                    orientation_map[0, j, i] = (objpos[0] - i) / output_res
                    orientation_map[1, j, i] = (objpos[1] - j) / output_res
                    orientation_weight_map[0, j, i] = 1
                    orientation_weight_map[1, j, i] = 1

        return orientation_map, orientation_weight_map
