import random
import colorsys
from PIL import Image
import cv2
import os
import numpy as np


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox_for_batch(image, bboxes, classes, show_label=True):
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

    for i, bbox in enumerate(bboxes):
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
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


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
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def save_detected_images(image_paths, bboxes, coco_label_path, output_dir='data/result'):
    """
    draw bbox on the origin image,and save batch detected result
    bboxes:[[  class_id, x1, x2, y1, y2, possibility], [  class, x1, x2, y1, y2, possibility]...]
    """
    assert len(image_paths) > 0 and os.path.exists(image_paths[0]) and len(image_paths) == len(bboxes)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classes = read_class_names(coco_label_path)
    for i in range(len(image_paths)):
        bgr_image = cv2.imread(image_paths[i])
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = draw_bbox_for_batch(rgb_image, bboxes[i], classes)
        image = Image.fromarray(image)
        # image.show()
        out_path = output_dir + os.sep + 'detected_' + os.path.basename(image_paths[i])
        image.save(out_path)  # 保存


def batch_postprocess_boxes_new(yolo_pos, yolo_prob, org_img_shape, threshold=0.3):
    """
    Generate batch bboxes for origin image based on network output,see >> postprocess_boxes()
    """
    batch_size, box_num, class_num = yolo_prob.shape[0], yolo_prob.shape[1], yolo_prob.shape[2] - 1
    batch_bboxes = []
    for i in range(batch_size):
        bboxes = postprocess_boxes_new(yolo_pos[i, :, :], yolo_prob[i, :, :], org_img_shape[i, :], threshold)
        bboxes = nms(bboxes, 0.45, method='nms')
        batch_bboxes.append(bboxes)
    return batch_bboxes


def batch_postprocess_boxes(yolo_pos, yolo_prob, org_img_shape, threshold=0.3):
    """
    Generate batch bboxes for origin image based on network output,see >> postprocess_boxes()
    """
    batch_size, class_num, box_num = yolo_prob.shape[0], yolo_prob.shape[1] - 1, yolo_prob.shape[2]
    batch_bboxes = []
    for i in range(batch_size):
        bboxes = postprocess_boxes(yolo_pos[i, :, :, :], yolo_prob[i, :, :], org_img_shape[i, :], threshold)
        batch_bboxes.append(bboxes)
    return batch_bboxes


def postprocess_boxes(yolo_pos, yolo_prob, org_img_shape, threshold=0.3):
    """
    Generate bboxes for origin image based on network output
    :param yolo_pos:   output bboxes with coordinates, shape(C+1, B, 4)   e.g.(81, 270, 4)
    :param yolo_prob:  output bboxes possibility,      shape(C+1, B)      e.g.(81, 270)
    :param org_img_shape: origin image shape, (height,width)
    :param threshold: a value to filter detected boxes, e.g. 0.5
    :return: bboxes:[[class_id, x1, x2, y1, y2, possibility], ... ,[class, x1, x2, y1, y2, possibility]]

    reference:
    https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/YOLOV3/core/utils.py
    """
    class_num = yolo_prob.shape[0] - 1
    box_num = yolo_prob.shape[1]
    pred_xywh = yolo_pos[1:yolo_pos.shape[1], :, 0:4]  # shape (B, C, 4)
    pred_xywh = pred_xywh.reshape((class_num * box_num, 4), order='F')  # shape (B*C, 4)
    pred_prob = yolo_prob[1:yolo_prob.shape[1], :].reshape((class_num * box_num), order='F')  # shape (B*C, )

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    org_h, org_w = org_img_shape
    xmin = (pred_xywh[:, 0] - pred_xywh[:, 2] / 2.) * org_w
    xmax = (pred_xywh[:, 0] + pred_xywh[:, 2] / 2.) * org_w
    ymin = (pred_xywh[:, 1] - pred_xywh[:, 3] / 2.) * org_h
    ymax = (pred_xywh[:, 1] + pred_xywh[:, 3] / 2.) * org_h
    pred_coor = np.vstack((xmin, ymin, xmax, ymax)).T

    # 2. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0  # shape (B*C, 4)

    # 3. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((0 < bboxes_scale), (bboxes_scale < np.inf))  # (B*C, )

    # 4. discard some boxes with pred_problow scores
    classes = np.tile(np.arange(class_num), box_num)
    score_mask = pred_prob > threshold
    mask = np.logical_and(scale_mask, score_mask)     # (B*C, )
    coors, pred_prob, classes = pred_coor[mask], pred_prob[mask], classes[mask]
    bboxes = np.concatenate([classes[:, np.newaxis], coors, pred_prob[:, np.newaxis]], axis=-1)
    return bboxes.tolist()


def postprocess_boxes_new(yolo_pos, yolo_prob, org_img_shape, threshold=0.3):
    """
    Generate bboxes for origin image based on network output
    :param yolo_pos:   output bboxes with coordinates, shape(box_num, 4)
    :param yolo_prob:  output bboxes possibility,      shape(box_num, class_num+1)
    :param org_img_shape: origin image height and width
    :param threshold: a value to filter detected boxes, e.g. 0.5
    :return: bboxes:[[class_id, x1, x2, y1, y2, possibility], ... ,[class, x1, x2, y1, y2, possibility]]

    reference:
    https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/YOLOV3/core/utils.py
    """
    print('yolo_pos.shape, yolo_prob.shape >>>>>>>>>>>>', yolo_pos.shape, yolo_prob.shape)
    pred_xywh = yolo_pos[:, 0:4]                     # shape (box_num, 4)
    pred_prob = yolo_prob[:, 1:yolo_prob.shape[1]]   # shape (box_num, 80)


    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    org_h, org_w = org_img_shape
    xmin = (pred_xywh[:, 0] - pred_xywh[:, 2] / 2.) * org_w
    xmax = (pred_xywh[:, 0] + pred_xywh[:, 2] / 2.) * org_w
    ymin = (pred_xywh[:, 1] - pred_xywh[:, 3] / 2.) * org_h
    ymax = (pred_xywh[:, 1] + pred_xywh[:, 3] / 2.) * org_h
    pred_coor = np.vstack((xmin, ymin, xmax, ymax)).T   # shape (box_num, 4)

    # 2. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0  # shape (box_num, 4)

    # 3. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((0 < bboxes_scale), (bboxes_scale < np.inf))  # (box_num, )

    # 4. discard some boxes with pred_problow scores
    classes = np.argmax(pred_prob, axis=-1)          # (box_num, )
    print('classes.shape .>>>>>>>>>>>>>>>>>>>>>>>', classes.shape)
    scores = pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > threshold
    mask = np.logical_and(scale_mask, score_mask)    # (box_num, )
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    # bboxes.shape >> (box_num, 6)
    bboxes = np.concatenate([classes[:, np.newaxis], coors, scores[:, np.newaxis]], axis=-1)
    return bboxes


def bboxes_iou(boxes1, boxes2):
    """IOU for box1 and box2"""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """Non-Maximum Suppression,NMS
    :param bboxes: shape >> (box_num, 6)   six dimension: [class_id, x1, x2, y1, y2, possibility]
    Note: soft-nms,
    https://arxiv.org/pdf/1704.04503.pdf
    https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 0]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 0] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 5])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, 1:5], cls_bboxes[:, 1:5])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


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
