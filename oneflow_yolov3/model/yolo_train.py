import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import math
from yolo_net import YoloTrainNet
import oneflow_yolov3
from oneflow_yolov3.ops.yolo_decode import yolo_train_decoder

parser = argparse.ArgumentParser(
    description="flags for multi-node and resource")
parser.add_argument("-load", "--model_load_dir", type=str, required=False)
parser.add_argument(
    "-gpu_num_per_node",
    "--gpu_num_per_node",
    type=int,
    default=1,
    required=False)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    default=1,
    required=False)
parser.add_argument(
    "-image_height",
    "--image_height",
    type=int,
    default=608,
    required=False)
parser.add_argument(
    "-image_width",
    "--image_width",
    type=int,
    default=608,
    required=False)
parser.add_argument(
    "-classes",
    "--classes",
    type=int,
    default=80,
    required=False)
parser.add_argument(
    "-total_images",
    "--total_images",
    type=int,
    default=117264,
    required=False)
parser.add_argument(
    "-num_boxes",
    "--num_boxes",
    type=int,
    default=90,
    required=False)
parser.add_argument("-hue", "--hue", type=float, default=0.1, required=False)
parser.add_argument(
    "-jitter",
    "--jitter",
    type=float,
    default=0.3,
    required=False)
parser.add_argument(
    "-saturation",
    "--saturation",
    type=float,
    default=1.5,
    required=False)
parser.add_argument(
    "-exposure",
    "--exposure",
    type=float,
    default=1.5,
    required=False)
parser.add_argument("-dataset_dir", "--dataset_dir", type=str, required=True)
parser.add_argument(
    "-num_epoch",
    "--num_epoch",
    type=int,
    default=10,
    required=False)
parser.add_argument(
    "-base_lr",
    "--base_lr",
    type=float,
    default=0.001,
    required=False)
parser.add_argument(
    "-save_frequency",
    "--save_frequency",
    type=int,
    required=True)
parser.add_argument("-save", "--model_save_dir", type=str, required=False)

args = parser.parse_args()

flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(args.base_lr)
func_config.train.model_update_conf(dict(naive_conf={}))


@flow.global_function("train", func_config)
def yolo_train_job():
    images, ground_truth, gt_valid_num = yolo_train_decoder(args.batch_size, args.image_height, args.image_width,
                                                            args.classes, args.num_boxes, args.hue, args.jitter, args.saturation, args.exposure, args.dataset_dir, "yolo")
    gt_boxes = flow.slice(
        ground_truth, [
            None, 0, 0], [
            None, -1, 4], name='gt_box')
    gt_labels = flow.cast(
        flow.slice(
            ground_truth, [
                None, 0, 4], [
                None, -1, 1], name='gt_label'), dtype=flow.int32)
    yolo_loss_result, statistics_info_result = YoloTrainNet(
        images, gt_boxes, gt_labels, gt_valid_num, True)
    flow.losses.add_loss(yolo_loss_result[0])
    flow.losses.add_loss(yolo_loss_result[1])
    flow.losses.add_loss(yolo_loss_result[2])
    return yolo_loss_result, statistics_info_result


def process_statistics_info(layer_name, statistics_info):
    count = statistics_info[:, 3].sum()
    class_count = statistics_info[:, 4].sum()
    avg_iou = statistics_info[:, 0].sum() / count if count != 0 else 0
    avg_recall_5 = statistics_info[:, 1].sum() / count if count != 0 else 0
    avg_recall_75 = statistics_info[:, 2].sum() / count if count != 0 else 0

    print(
        layer_name,
        "Avg IOU: ",
        avg_iou,
        ".5 Recall:",
        avg_recall_5,
        ".75 Recall:",
        avg_recall_75,
        "count:",
        count)


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}  {:>12.4f}  {:>12.4f} {:>12.4f} {:>12.3f}"
    print(
        "{:>12}      {:>12}      {:>12}".format(
            "iter",
            "losses value",
            "time"))
    global cur_time
    cur_time = time.time()

    iter_num = math.ceil(args.total_images / args.batch_size)
    for epoch in range(args.num_epoch):
        for iter in range(iter_num):
            yolo_loss_result, statistics_info_result = yolo_train_job().get()
            process_statistics_info(
                "Region 82", statistics_info_result[0].numpy())
            process_statistics_info(
                "Region 94", statistics_info_result[1].numpy())
            process_statistics_info(
                "Region 106", statistics_info_result[2].numpy())
            print(
                fmt_str.format(
                    iter, np.abs(
                        yolo_loss_result[0]).mean(), np.abs(
                        yolo_loss_result[1]).mean(), np.abs(
                        yolo_loss_result[2]).mean(), time.time() - cur_time))
            cur_time = time.time()
        if (epoch + 1) % args.save_frequency == 0:
            check_point.save(args.model_save_dir +
                             "/yolov3_snapshot_" + str(epoch + 1))
