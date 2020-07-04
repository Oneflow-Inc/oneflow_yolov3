import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloPredictNet
from data_preprocess_multiprocess import make_work
import oneflow_yolov3
from oneflow_yolov3.ops.yolo_decode import yolo_predict_decoder

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

label_2_name=[]

with open(args.label_to_name_file,'r') as f:
  label_2_name=f.readlines()

nms = True


def print_detect_box(positions, probs):
    if nms==True:
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
    x1  = (x - w / 2.) * origin_image[1]
    x2  = (x + w / 2.) * origin_image[1]
    y1   = (y - h / 2.) * origin_image[0]
    y2   = (y + h / 2.) * origin_image[0]
    return x1, y1, x2, y2


def batch_boxes(positions, probs, origin_image_info):
    batch_size = positions.shape[0]
    batch_list=[]
    if nms==True:
        for k in range(batch_size):
            box_list = []
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][2]!=0 and positions[k][i][j][3]!=0 and probs[k][i][j]!=0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][i][j][0], positions[k][i][j][1], positions[k][i][j][2], positions[k][i][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][i][j]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    else:
        for k in range(batch_size):
            box_list = []
            for j in range(positions.shape[1]):
                for i in range(1, 81):
                    if positions[k][j][2]!=0 and positions[k][j][3]!=0 and probs[k][j][i]!=0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][j][0], positions[k][j][1], positions[k][j][2], positions[k][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][j][i]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    return batch_list


input_blob_def_dict = {
    "images" : flow.FixedTensorDef((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float),
    "origin_image_info" : flow.FixedTensorDef((args.batch_size, 2), dtype=flow.int32),
}


def yolo_user_op_eval_job(images=input_blob_def_dict["images"], origin_image_info=input_blob_def_dict["origin_image_info"]):
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    return yolo_pos_result, yolo_prob_result, origin_image_info


if __name__ == "__main__":
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.ctrl_port(9789)

    check_point = flow.train.CheckPoint()
    if not args.model_load_dir:
        check_point.init()
    else:
        check_point.load(args.model_load_dir)
    fmt_str = "{:>12}   {:>12.10f} {:>12.10f} {:>12.3f}"
    print("{:>12}   {:>12}  {:>12}  {:>12}".format("iter",  "reg loss value", "cls loss value", "time"))
    global cur_time
    cur_time = time.time()

    num_worker=4
    queue_size=args.batch_size//num_worker
    image_height = 608
    image_width = 608
    workers = []
    for i in range(num_worker):
        workers.append(make_work(i))
    for w in workers:
        w[0].start()

    batch_image = np.zeros((num_worker, queue_size*3*image_height*image_width), dtype=np.float32)
    batch_origin_info = np.zeros((num_worker, queue_size*2), dtype=np.int32)

    image_list_len = len(args.image_path_list)
    for step in range(args.total_batch_num):
        #img_path = args.image_path_list[step % image_list_len]
        path=['data/images/00000'+str(step+1)+'.jpg'] * 16 
        for i in range(num_worker):
            _, _, w_q, _, _ = workers[i]
            for j in range(queue_size):
                w_q.put(path[i * queue_size + j])
        
        for i in range(num_worker):
            (proc, w_f, _, w_b, w_origin_info_b) = workers[i]
            while w_f.value == 0:
                time.sleep(0.01)
            ret = np.ctypeslib.as_array(w_b)
            batch_image[i,:] = ret
            ret_origin_info = np.ctypeslib.as_array(w_origin_info_b)
            batch_origin_info[i,:] = ret_origin_info
            w_f.value = 0
        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(batch_image.reshape(args.batch_size, 3, image_height, image_width), batch_origin_info.reshape(args.batch_size, 2)).get()
        batch_list = batch_boxes(yolo_pos, yolo_prob, origin_image_info)
        print("batch_list", batch_list)
    
    for w in workers:
        w[0].terminate()

    for w in workers:
        w[0].join()
