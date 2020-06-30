from multiprocessing import Process, Lock, Queue
from multiprocessing import Value, Array, RawArray
from ctypes import Structure, c_double, memmove
import sys
sys.path.remove("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np 
import time

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
    for img_path in img_path_list:
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


img_path_list=[]
image_height = 608
image_width = 608
batch_size = 16
num_worker = 4
queue_size = batch_size // num_worker

size = queue_size * 3 * image_height * image_width
origin_info_size = queue_size * 2

def worker(idx, flag, buffer, origin_info_buffer, queue):
    while True:
        time1 = time.time()
        img_path_list = []
        for i in range(queue_size):
            path = queue.get()
            img_path_list.append(path)
        images, origin_image_info = batch_image_preprocess_v2(img_path_list, image_width, image_width)
        while flag.value != 0:
            time.sleep(0.01)
        arr = np.ctypeslib.as_array(buffer)
        arr[:] = images.flatten()
        origin_info_arr = np.ctypeslib.as_array(origin_info_buffer)
        origin_info_arr[:] = origin_image_info.flatten()

        flag.value = 1
        #print("make data time", time.time()-time1)

def make_work(idx):
    flag = Value('i', 0, lock=True)
    buffer = RawArray('f', size)
    origin_info_buffer = RawArray('i', origin_info_size)
    queue = Queue(queue_size)
    proc = Process(target=worker, args=(idx, flag, buffer, origin_info_buffer, queue))
    return proc, flag, queue, buffer, origin_info_buffer

#if __name__ == '__main__':
#    workers = []
#    for i in range(num_worker):
#        workers.append(make_work(i))
#    for w in workers:
#        w[0].start()
#
#    ss = time.time()
#
#    for it in range(1, 5):
#        path=['../../data/images/00000'+str(it)+'.jpg'] * 16 
#        for i in range(num_worker):
#            _, _, w_q, _, _ = workers[i]
#            for j in range(queue_size):
#                w_q.put(path[i * queue_size + j])
#        
#        s = time.time()
#        batch_image = np.zeros((num_worker, queue_size*3*image_height*image_width))
#        batch_origin_info = np.zeros((num_worker, queue_size*2))
#        for i in range(num_worker):
#            (proc, w_f, _, w_b, w_origin_info_b) = workers[i]
#            while w_f.value == 0:
#                time.sleep(0.01)
#            ret = np.ctypeslib.as_array(w_b)
#            batch_image[i,:] = ret
#            ret_origin_info = np.ctypeslib.as_array(w_origin_info_b)
#            batch_origin_info[i,:] = ret_origin_info
#            w_f.value = 0
#
#        e = time.time()
#        print(it, e - s, e)
#
#    for w in workers:
#        w[0].terminate()
#
#    for w in workers:
#        w[0].join()

