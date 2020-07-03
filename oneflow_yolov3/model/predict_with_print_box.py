import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloPredictNet
import oneflow_yolov3
from oneflow_yolov3.ops.yolo_decode import yolo_predict_decoder

import random
import colorsys
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, required=True)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-img_list", "--image_list_path", type=str, required=True)
parser.add_argument("-label_to_name_file", "--label_to_name_file", type=str, required=True)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, default=308, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-use_tensorrt", "--use_tensorrt", type=int, default=0, required=False)
parser.add_argument("-image_path_list", "--image_path_list", type=str, nargs='+', required=True)

args = parser.parse_args()

assert os.path.exists(args.model_load_dir)
assert os.path.exists(args.image_path_list[0])
assert os.path.exists(args.label_to_name_file)


flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)
#func_config.tensorrt.use_fp16()

label_2_name = []
nms = True
with open(args.label_to_name_file, 'r') as f:
    label_2_name = f.readlines()


def print_detect_box(positions, probs):
    if nms:
        batch_size = positions.shape[0]
        for k in range(batch_size):
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][1]!=0 and positions[k][i][j][2]!=0 and probs[k][i][j]!=0:
                        print(label_2_name[i-1], " ", probs[k][i][j]*100,"%", "  ", positions[k][i][j][0], " ", positions[k][i][j][1], " ", positions[k][i][j][2], " ", positions[k][i][j][3])
    else:
        for j in range(positions.shape[1]):
            for i in range(1, 81):
                if positions[0][j][1]!=0 and positions[0][j][2]!=0 and probs[0][j][i]!=0:
                    print(label_2_name[i-1], " ", probs[0][j][i]*100,"%", "  ",positions[0][j][0], " ", positions[0][j][1], " ", positions[0][j][2], " ", positions[0][j][3])


def xywh_2_x1y1x2y2(x, y, w, h, origin_image):
    x1 = (x - w / 2.) * origin_image[1]
    x2 = (x + w / 2.) * origin_image[1]
    y1 = (y - h / 2.) * origin_image[0]
    y2 = (y + h / 2.) * origin_image[0]
    return x1, y1, x2, y2


def batch_boxes(positions, probs, origin_image_info):
    batch_size = positions.shape[0]
    batch_list = []
    if nms:
        for k in range(batch_size):
            box_list = []
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][2] != 0 and positions[k][i][j][3] != 0 and probs[k][i][j] != 0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][i][j][0], positions[k][i][j][1],
                                                         positions[k][i][j][2], positions[k][i][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][i][j]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    else:
        for k in range(batch_size):
            box_list = []
            for j in range(positions.shape[1]):
                for i in range(1, 81):
                    if positions[k][j][2] != 0 and positions[k][j][3] != 0 and probs[k][j][i] != 0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][j][0], positions[k][j][1],
                                                         positions[k][j][2], positions[k][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][j][i]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    return batch_list


def striplist(l):
    return [x.strip() for x in l]


@flow.global_function(func_config)
def yolo_user_op_eval_job():
    images, origin_image_info = yolo_predict_decoder(args.batch_size, args.image_height,
                                                     args.image_width, args.image_path_list, "yolo")
    images = flow.identity(images, name="yolo-layer1-start")
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    yolo_pos_result = flow.identity(yolo_pos_result, name="yolo_pos_result_end")
    yolo_prob_result = flow.identity(yolo_prob_result, name="yolo_prob_result_end")
    return yolo_pos_result, yolo_prob_result, origin_image_info


def draw_and_save_detected_result(image_path, bboxes):
    """
    draw bbox on the origin image,and save detected result
    bboxes:[[  class, x1, x2, y1, y2, possibility], [  class, x1, x2, y1, y2, possibility]...]
    """
    assert os.path.isfile(image_path)
    if len(bboxes) == 1 and len(bboxes[0] > 0):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = draw_bbox(rgb_image, bboxes)
        image = Image.fromarray(image)
        # image.show()
        pts = os.path.splitext(image_path)
        image.save(pts[0] + '_detected' + pts[1])  # 保存


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes=read_class_names(args.label_to_name_file), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    for i, bbox in enumerate(bboxes[0]):
        if len(bbox) < 5:
            continue
        coor = np.array(bbox[1:5], dtype=np.int32)
        fontScale = 0.5
        score = bbox[5]
        class_ind = int(bbox[0])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.ctrl_port(9789)
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    images_path_list = args.image_path_list
    # initial network

    for i in range(len(images_path_list)):
        start = time.time()
        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job().get()
        end = time.time()
        bboxes = batch_boxes(yolo_pos, yolo_prob, origin_image_info)
        draw_and_save_detected_result(images_path_list[i], bboxes)
        print('%s >>> bboxes:' % images_path_list[i], bboxes)
        print('cost time: %.4f ms\n--------------------------------------------------------------'
              % (1000 * (end-start)))
