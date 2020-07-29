import argparse
import math
import os

import numpy as np
import oneflow as flow
import oneflow.typing as tp
import utils
from data_preprocess import batch_image_preprocess_with_label
from tqdm import tqdm
from yolo_net import YoloPredictNet

import oneflow_yolov3

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument(
    "-g",
    "--gpu_num_per_node",
    type=int,
    default=1,
    required=False)
parser.add_argument("-load", "--model_load_dir", type=str, required=True)
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
parser.add_argument("-label_path", "--label_path", type=str, required=True)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    default=8,
    required=False)
parser.add_argument('--iou_thres', type=float, default=0.5,
                    help='iou threshold required to qualify as detected')
parser.add_argument(
    '--conf_thres',
    type=float,
    default=0.3,
    help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5,
                    help='iou threshold for non-maximum suppression')
parser.add_argument(
    "-use_tensorrt",
    "--use_tensorrt",
    type=int,
    default=0,
    required=False)
parser.add_argument("-image_paths", "--image_paths", type=str, required=False)
args = parser.parse_args()


flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_logical_view(flow.scope.consistent_view())
func_config.default_data_type(flow.float)
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)
# func_config.tensorrt.use_fp16()


@flow.global_function("predict", func_config)
def yolo_user_op_eval_job(images:tp.Numpy.Placeholder((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float),
                          origin_image_info:tp.Numpy.Placeholder((args.batch_size, 2), dtype=flow.int32)
                          ):
    yolo_pos_result, yolo_prob_result = YoloPredictNet(
        images, origin_image_info, trainable=False)
    return yolo_pos_result, yolo_prob_result, origin_image_info


if __name__ == "__main__":
    assert os.path.exists(args.model_load_dir) and os.path.exists(args.image_paths) and os.path.exists(args.label_path)

    print('Params >> nms_thres: %.4f, conf_thres: %.4f, iou_thres: %.4f\n' %(args.nms_thres, args.conf_thres, args.iou_thres))

    with open(args.label_path, 'r') as f:
        names = f.read().split('\n')
    names = list(filter(None, names))
    flow.config.gpu_device_num(args.gpu_num_per_node)
    # Load model
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    path_list = []
    with open(args.image_paths) as f:
        for line in f:
            path_list.append(line.strip('\n'))

    iter_num = math.floor(len(path_list) / float(args.batch_size))

    # evaluate mAP
    """
    reference:
    https://github.com/ultralytics/yolov3/blob/master/test.py
    """
    seen = 0
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for i in tqdm(range(iter_num)):
        paths = path_list[i * args.batch_size:(i + 1) * args.batch_size]
        images, origin_image_info, targets = batch_image_preprocess_with_label(
            paths, args.image_height, args.image_width)

        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(
            images, origin_image_info).get()
        bboxes = utils.batch_postprocess_boxes_nms(
            yolo_pos, yolo_prob, origin_image_info, args.conf_thres, args.nms_thres)

        for si, pred in enumerate(bboxes):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            pred = np.asarray(pred).astype(np.float32)
            if len(pred) == 0:
                if nl:
                    stats.append(([], [], [], tcls))
                continue

            # Append to pycocotools JSON dictionary
            # Clip boxes to image bounds
            pred = utils.clip_coords(pred, origin_image_info[si])

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = labels[:, 1:5]

                # Search for correct predictions
                for i, p in enumerate(pred):
                    pcls = p[0]
                    pbox = p[1:5]

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls not in tcls:
                        continue
                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor)
                    m = np.nonzero(np.asarray(m))[0]
                    iou = utils.bboxes_iou(pbox, tbox[m])
                    bi = np.argmax(iou)
                    maxiou = iou[bi]

                    # If iou > threshold and class is correct mark as correct
                    # if maxiou > args.iou_thres:  # and pcls == tcls[bi]:
                    if maxiou > args.iou_thres and m[bi] not in detected:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 5], pred[:, 0], tcls))
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = utils.ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64),
                         minlength=80)  # number of targets per class

    print(('%20s' + '%10s' * 6) %
          ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1'))
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
