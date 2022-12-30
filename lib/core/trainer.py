# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from utils.utils import AverageMeter
from utils.vis import save_debug_images


def do_train(
    cfg,
    model,
    data_loader,
    loss_factory,
    optimizer,
    epoch,
    output_dir,
    tb_log_dir,
    writer_dict,
    fp16=False,
):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    up_conf_loss_meter = AverageMeter()
    down_conf_loss_meter = AverageMeter()
    up_orie_loss_meter = AverageMeter()
    down_orie_loss_meter = AverageMeter()
    up_to_down_orie_loss_meter = AverageMeter()
    down_to_up_orie_loss_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, heatmaps, masks, joints, daes) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(images.cuda())

        # heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        # masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        # daes = list(map(lambda x: x.cuda(non_blocking=True), daes))

        # loss = loss_factory(outputs, heatmaps, masks)
        up_conf_loss, down_conf_loss, up_orie_loss, down_orie_loss, up_to_down_orie_loss, down_to_up_orie_loss \
             = loss_factory(outputs, heatmaps, masks, daes)

        up_conf_loss = up_conf_loss.mean(dim=0)
        up_conf_loss_meter.update(up_conf_loss.item(), images.size(0))
        down_conf_loss = down_conf_loss.mean(dim=0)
        down_conf_loss_meter.update(down_conf_loss.item(), images.size(0))
        up_orie_loss = up_orie_loss.mean(dim=0)
        up_orie_loss_meter.update(up_orie_loss.item(), images.size(0))
        down_orie_loss = down_orie_loss.mean(dim=0)
        down_orie_loss_meter.update(down_orie_loss.item(), images.size(0))
        up_to_down_orie_loss = up_to_down_orie_loss.mean(dim=0)
        up_to_down_orie_loss_meter.update(up_to_down_orie_loss.item(), images.size(0))
        down_to_up_orie_loss = down_to_up_orie_loss.mean(dim=0)
        down_to_up_orie_loss_meter.update(down_to_up_orie_loss.item(), images.size(0))


        loss = up_conf_loss+down_conf_loss+ \
            up_orie_loss+down_orie_loss+ \
            up_to_down_orie_loss+down_to_up_orie_loss
        

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        up_conf_loss, down_conf_loss, up_orie_loss, down_orie_loss, up_to_down_orie_loss, down_to_up_orie_loss

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed: {speed:.1f} samples/s\t"
                "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "{up_conf_loss}{down_conf_loss}{up_orie_loss}{down_orie_loss}{up_to_down_orie_loss}{down_to_up_orie_loss}".format(
                    epoch,
                    i,
                    len(data_loader),
                    batch_time=batch_time,
                    speed=images.size(0) / batch_time.val,
                    data_time=data_time,
                    up_conf_loss=_get_loss_info(up_conf_loss_meter, "up_conf_loss"),
                    down_conf_loss=_get_loss_info(down_conf_loss_meter, "down_conf_loss"),
                    up_orie_loss=_get_loss_info(up_orie_loss_meter, "up_orie_loss"),
                    down_orie_loss=_get_loss_info(down_orie_loss_meter, "down_orie_loss"),
                    up_to_down_orie_loss=_get_loss_info(up_to_down_orie_loss_meter, "up_to_down_orie_loss"),
                    down_to_up_orie_loss=_get_loss_info(down_to_up_orie_loss_meter, "down_to_up_orie_loss")
                )
            )
            logger.info(msg)

            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar(
                "up_conf_loss",
                up_conf_loss_meter.val,
                global_steps,
            )
            writer.add_scalar(
                "down_conf_loss",
                down_conf_loss_meter.val,
                global_steps,
            )
            writer.add_scalar(
                "up_orie_loss",
                up_orie_loss_meter.val,
                global_steps,
            )
            writer.add_scalar(
                "down_orie_loss",
                down_orie_loss_meter.val,
                global_steps,
            )
            writer.add_scalar(
                "up_conf_loss",
                up_to_down_orie_loss_meter.val,
                global_steps,
            )
            writer.add_scalar(
                "down_to_up_orie_loss",
                down_to_up_orie_loss_meter.val,
                global_steps,
            )

            writer_dict["train_global_steps"] = global_steps + 1

            # prefix = "{}_{}".format(os.path.join(output_dir, "train"), i)
            # for scale_idx in range(len(outputs)):
            #     prefix_scale = prefix + "_output_{}".format(
            #         cfg.DATASET.OUTPUT_SIZE[scale_idx]
            #     )
            #     save_debug_images(
            #         cfg,
            #         images,
            #         heatmaps[scale_idx],
            #         masks[scale_idx],
            #         outputs[scale_idx],
            #         prefix_scale,
            #     )


def _get_loss_info(loss_meter, loss_name):
    msg = "{name}: {meter.val:.3e} ({meter.avg:.3e})\t".format(
        name=loss_name, meter=loss_meter
    )
    return msg
