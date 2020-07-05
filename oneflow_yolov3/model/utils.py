import random
import colorsys
from PIL import Image
import cv2
import os
import numpy as np


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes, show_label=True):
    """
    bbox: [cls_id, x_min, y_min, x_max, y_max, probability] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    # print('bboxes, bboxes[0] >>>>>>>>>>>>>>>>', bboxes, bboxes[0])
    for i, bbox in enumerate(bboxes[0]):
        # print('bbox >>>>>>>>>>>>>>>>', bbox)
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


def save_detected_result(image_path, bboxes, coco_label_path):
    """
    draw bbox on the origin image,and save detected result
    bboxes:[[  class_id, x1, x2, y1, y2, possibility], [  class, x1, x2, y1, y2, possibility]...]
    """
    assert os.path.isfile(image_path)
    classes = read_class_names(coco_label_path)
    if len(bboxes) == 1 and len(bboxes[0]) > 0:
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = draw_bbox(rgb_image, bboxes, classes)
        image = Image.fromarray(image)
        # image.show()
        pts = os.path.splitext(image_path)
        image.save(pts[0] + '_detected' + pts[1])  # 保存


def postprocess_boxes(yolo_pos, yolo_prob, org_img_shape, threshold):
    """
    Generate bboxes for origin image based on network output
    :param yolo_pos:   output bboxes with coordinates, shape(1, C+1, B, 4)   e.g.(1, 81, 270, 4)
    :param yolo_prob:  output bboxes possibility, shape(1, C+1, B)           e.g.(1, 81, 270)
    :param org_img_shape: origin image shape, (height,width)
    :param threshold: a value to filter detected boxes, e.g. 0.5
    :return: bboxes:[[class_id, x1, x2, y1, y2, possibility], [class, x1, x2, y1, y2, possibility]...]
    """
    class_num = yolo_prob.shape[1]-1
    box_num = yolo_prob.shape[2]
    pred_xywh = yolo_pos[0, 1:yolo_pos.shape[1], :, 0:4]                            # shape (B, C, 4)
    pred_xywh = pred_xywh.reshape((class_num * box_num, 4), order='F')              # shape (B*C, 4)
    pred_prob = yolo_prob[0, 1:yolo_prob.shape[1], :].reshape((class_num*box_num), order='F')  # shape (B*C, )

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    org_h, org_w = org_img_shape
    xmin = (pred_xywh[:, 0] - pred_xywh[:, 2] / 2.) * org_w
    xmax = (pred_xywh[:, 0] + pred_xywh[:, 2] / 2.) * org_w
    ymin = (pred_xywh[:, 1] - pred_xywh[:, 3] / 2.) * org_h
    ymax = (pred_xywh[:, 1] + pred_xywh[:, 3] / 2.) * org_h
    pred_coor = np.vstack((xmin, ymin, xmax, ymax)).T

    # 2. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),  # shape (B*C, 4)
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 3. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((0 < bboxes_scale), (bboxes_scale < np.inf))  # (B*C, )

    # 4. discard some boxes with pred_problow scores
    classes = np.tile(np.arange(class_num), box_num)   # (B*C, )
    score_mask = pred_prob > threshold                 # (B*C, )
    mask = np.logical_and(scale_mask, score_mask)
    coors, pred_prob, classes = pred_coor[mask], pred_prob[mask], classes[mask]
    bboxes = np.concatenate([classes[:, np.newaxis], coors, pred_prob[:, np.newaxis]], axis=-1)
    return [bboxes.tolist()]


def xywh_2_x1y1x2y2(x, y, w, h, origin_image):
    """
    [depreciated]
    fixed size image coordinates to origin image coordinate
    """
    x1 = (x - w / 2.) * origin_image[1]
    x2 = (x + w / 2.) * origin_image[1]
    y1 = (y - h / 2.) * origin_image[0]
    y2 = (y + h / 2.) * origin_image[0]
    return x1, y1, x2, y2


def batch_boxes(positions, probs, origin_image_info, nms=True):
    """
    [depreciated]
    positions.shape >>   (batch, 81, 270, 4)
    probs.shape >>       (batch, 81, 270)
    origin_image_info >> [height, width]
    """
    batch_size = positions.shape[0]
    batch_list = []
    if nms:
        for k in range(batch_size):
            box_list = []
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][2] != 0 and positions[k][i][j][3] != 0 and probs[k][i][j] != 0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][i][j][0], positions[k][i][j][1],
                                                         positions[k][i][j][2], positions[k][i][j][3],
                                                         origin_image_info[k])
                        bbox = [i - 1, x1, y1, x2, y2, probs[k][i][j]]
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
                        bbox = [i - 1, x1, y1, x2, y2, probs[k][j][i]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    return batch_list