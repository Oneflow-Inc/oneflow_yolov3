import sys
import cv2
import os
import numpy as np 
import time

def image_preprocess(img_path, image_height, image_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 

    w = image_width 
    h = image_height
    origin_h = img.shape[0]
    origin_w = img.shape[1]
    new_w = origin_w
    new_h = origin_h
    if w/origin_w < h/origin_h:
        new_w = w 
        new_h = origin_h * w // origin_w
    else:
        new_h = h 
        new_w = origin_w * h // origin_h    

    resize_img = cv2.resize(img,(int(new_w),int(new_h)), interpolation=cv2.INTER_CUBIC)
    resize_img = resize_img.transpose(2, 0, 1).astype(np.float32)
    resize_img = resize_img / 255
    resize_img[[0,1,2], :, :] = resize_img[[2,1,0], :, :]

    dw = (w-new_w)//2
    dh = (h-new_h)//2

    padh_before = int(dh)
    padh_after = int(h - new_h - padh_before)
    padw_before = int(dw)
    padw_after = int(w - new_w - padw_before)
    result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
    result = np.expand_dims(result, axis=0)
    origin_image_info = np.zeros((1,2), dtype=np.int32)
    origin_image_info[0,0] = origin_h
    origin_image_info[0,1] = origin_w
    return result, origin_image_info


def batch_image_preprocess(img_path_list, image_height, image_width):
    result_list = []
    origin_info_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 

        w = image_width 
        h = image_height
        origin_h = img.shape[0]
        origin_w = img.shape[1]
        new_w = origin_w
        new_h = origin_h
        if w/origin_w < h/origin_h:
            new_w = w 
            new_h = origin_h * w // origin_w
        else:
            new_h = h 
            new_w = origin_w * h // origin_h    

        resize_img = cv2.resize(img,(int(new_w),int(new_h)), interpolation=cv2.INTER_CUBIC)
        resize_img = resize_img.transpose(2, 0, 1).astype(np.float32)
        resize_img = resize_img / 255
        resize_img[[0,1,2], :, :] = resize_img[[2,1,0], :, :]

        dw = (w-new_w)//2
        dh = (h-new_h)//2

        padh_before = int(dh)
        padh_after = int(h - new_h - padh_before)
        padw_before = int(dw)
        padw_after = int(w - new_w - padw_before)
        result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
        origin_image_info = [origin_h, origin_w]
        result_list.append(result)
        origin_info_list.append(origin_image_info)
    results = np.asarray(result_list).astype(np.float32)
    origin_image_infos = np.asarray(origin_info_list).astype(np.int32)
    return results, origin_image_infos


def resize_image(img, origin_h, origin_w, image_height, image_width):
    w = image_width
    h = image_height
    resized=np.zeros((3, image_height, image_width), dtype=np.float32)
    part=np.zeros((3, origin_h, image_width), dtype = np.float32)
    w_scale = (float)(origin_w - 1) / (w - 1)
    h_scale = (float)(origin_h - 1) / (h - 1)

    for c in range(w):
        if c == w-1 or origin_w == 1:
            val = img[:, :, origin_w-1]
        else:
            sx = c * w_scale
            ix = int(sx)
            dx = sx - ix 
            val = (1 - dx) * img[:, :, ix] + dx * img[:, :, ix+1]
        part[:, :, c] = val
    for r in range(h):
        sy = r * h_scale
        iy = int(sy)
        dy = sy - iy
        val = (1-dy)*part[:, iy, :]
        resized[:, r, :] = val 
        if r==h-1 or origin_h==1:
            continue
        resized[:, r, :] = resized[:, r, :] + dy * part[:, iy+1, :]
    return resized



def image_preprocess_v2(img_path, image_height, image_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    img = img.transpose(2, 0, 1).astype(np.float32) # hwc->chw
    img = img / 255  # /255
    img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

    w = image_width 
    h = image_height
    origin_h = img.shape[1]
    origin_w = img.shape[2]
    new_w = origin_w
    new_h = origin_h
    if w/origin_w < h/origin_h:
        new_w = w 
        new_h = origin_h * w // origin_w
    else:
        new_h = h 
        new_w = origin_w * h // origin_h    
    resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

    dw = (w-new_w)//2
    dh = (h-new_h)//2

    padh_before = int(dh)
    padh_after = int(h - new_h - padh_before)
    padw_before = int(dw)
    padw_after = int(w - new_w - padw_before)
    result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
    result = np.expand_dims(result, axis=0)
    origin_image_info = np.zeros((1,2), dtype=np.int32)
    origin_image_info[0,0] = origin_h
    origin_image_info[0,1] = origin_w
    return result.astype(np.float32), origin_image_info

def batch_image_preprocess_v2(img_path_list, image_height, image_width):
    result_list = []
    origin_info_list = []
    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = img.transpose(2, 0, 1).astype(np.float32) # hwc->chw
        img = img / 255  # /255
        img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

        w = image_width 
        h = image_height
        origin_h = img.shape[1]
        origin_w = img.shape[2]
        new_w = origin_w
        new_h = origin_h
        if w/origin_w < h/origin_h:
            new_w = w 
            new_h = origin_h * w // origin_w
        else:
            new_h = h 
            new_w = origin_w * h // origin_h    
        resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

        dw = (w-new_w)//2
        dh = (h-new_h)//2

        padh_before = int(dh)
        padh_after = int(h - new_h - padh_before)
        padw_before = int(dw)
        padw_after = int(w - new_w - padw_before)
        result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
        origin_image_info = [origin_h, origin_w]
        result_list.append(result)
        origin_info_list.append(origin_image_info)
    results = np.asarray(result_list).astype(np.float32)
    origin_image_infos = np.asarray(origin_info_list).astype(np.int32)
    return results, origin_image_infos


def batch_image_preprocess_with_label(img_path_list, image_height, image_width):
    result_list = []
    origin_info_list = []
    labels = np.empty((0, 6), np.float32)
    for i, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = img.transpose(2, 0, 1).astype(np.float32) # hwc->chw
        img = img / 255  # /255
        img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

        w = image_width 
        h = image_height
        origin_h = img.shape[1]
        origin_w = img.shape[2]
        new_w = origin_w
        new_h = origin_h
        if w/origin_w < h/origin_h:
            new_w = w 
            new_h = origin_h * w // origin_w
        else:
            new_h = h 
            new_w = origin_w * h // origin_h    
        resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

        dw = (w-new_w)//2
        dh = (h-new_h)//2

        padh_before = int(dh)
        padh_after = int(h - new_h - padh_before)
        padw_before = int(dw)
        padw_after = int(w - new_w - padw_before)
        result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
        origin_image_info = [origin_h, origin_w]
        result_list.append(result)
        origin_info_list.append(origin_image_info)
        label_path = img_path.split('.jpg')[0] + '.txt'
        label = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                label = x.copy()
                label[:, 1] = origin_w * (x[:, 1] - x[:, 3] / 2)
                label[:, 2] = origin_h * (x[:, 2] - x[:, 4] / 2)
                label[:, 3] = origin_w * (x[:, 1] + x[:, 3] / 2)
                label[:, 4] = origin_h * (x[:, 2] + x[:, 4] / 2)

        nL = len(label)
        if nL:
            ind = np.full((nL, 1), i)
            label = np.append(ind, label, axis=1)
            labels = np.append(labels, label, axis=0)

   
    results = np.asarray(result_list).astype(np.float32)
    origin_image_infos = np.asarray(origin_info_list).astype(np.int32)
    return results, origin_image_infos, labels
