import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloTrainNet
import oneflow_yolov3
from oneflow_yolov3.ops.yolo_decode import yolo_train_decoder

parser = argparse.ArgumentParser(description="flags for multi-node and resource")
parser.add_argument("-load", "--model_load_dir", type=str, required=False)
parser.add_argument("-gpu_num_per_node", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-classes", "--classes", type=int, default=80, required=False)
parser.add_argument("-num_boxes", "--num_boxes", type=int, default=90, required=False)
parser.add_argument("-hue", "--hue", type=float, default=0.1, required=False)
parser.add_argument("-jitter", "--jitter", type=float, default=0.3, required=False)
parser.add_argument("-saturation", "--saturation", type=float, default=1.5, required=False)
parser.add_argument("-exposure", "--exposure", type=float, default=1.5, required=False)
parser.add_argument("-image_path_file", "--image_path_file", type=str, required=True)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, default=10, required=False)
parser.add_argument("-base_lr", "--base_lr", type=float, default=0.001, required=False)


args = parser.parse_args()

flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
func_config.train.primary_lr(0.001)
func_config.train.model_update_conf(dict(naive_conf={}))

@flow.global_function(func_config)
def yolo_train_job():
    images, ground_truth, gt_valid_num = yolo_train_decoder(args.batch_size, args.image_height, args.image_width, args.classes, args.num_boxes, args.hue, args.jitter, args.saturation, args.exposure, args.image_path_file, "yolo")
    gt_boxes = flow.slice(ground_truth, [None, 0, 0], [None, -1, 4], name = 'gt_box')
    gt_labels = flow.cast(flow.slice(ground_truth, [None, 0, 4], [None, -1, 1], name = 'gt_label'), dtype=flow.int32)
    yolo0_loss, yolo1_loss, yolo2_loss = YoloTrainNet(images, gt_boxes, gt_labels, gt_valid_num, True)
    flow.losses.add_loss(yolo0_loss)
    flow.losses.add_loss(yolo1_loss)
    flow.losses.add_loss(yolo2_loss)
    return yolo0_loss, yolo1_loss, yolo2_loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}   {:>12.10f} {:>12.10f} {:>12.3f}"
    print("{:>12}   {:>12}  {:>12}  {:>12}".format("iter",  "reg loss value", "cls loss value", "time"))
    global cur_time
    cur_time = time.time()


    for step in range(args.total_batch_num):
        yolo0_loss, yolo1_loss, yolo2_loss = yolo_train_job().get()
        print(yolo0_loss.mean())
