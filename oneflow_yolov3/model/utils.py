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


def draw_and_save_detected_result(image_path, bboxes, coco_label_path):
    """
    draw bbox on the origin image,and save detected result
    bboxes:[[  class, x1, x2, y1, y2, possibility], [  class, x1, x2, y1, y2, possibility]...]
    """
    assert os.path.isfile(image_path)
    classes = read_class_names(coco_label_path)
    if len(bboxes) == 1 and len(bboxes[0] > 0):
        bgr_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = draw_bbox(rgb_image, bboxes, classes)
        image = Image.fromarray(image)
        # image.show()
        pts = os.path.splitext(image_path)
        image.save(pts[0] + '_detected' + pts[1])  # 保存