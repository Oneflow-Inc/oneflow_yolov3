import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import time
import argparse
import numpy as np
import os
from yolo_net import YoloPredictNet
import yolov3
from yolov3.ops.yolo_decode import yolo_predict_decoder

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


args = parser.parse_args()

assert os.path.exists(args.model_load_dir)
assert os.path.exists(args.image_list_path)
assert os.path.exists(args.label_to_name_file)
with open(args.image_list_path, 'r') as f:
    image_paths=f.read()
    assert os.path.exists(image_paths.splitlines()[0])


flow.config.load_library(yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)
#func_config.tensorrt.use_fp16()

label_2_name=[]

with open(args.label_to_name_file,'r') as f:
  label_2_name=f.readlines()

nms=False

print("nms:", nms)
def print_detect_box(positions, probs):
    if nms==True:
        batch_size = positions.shape[0]
        for k in range(batch_size):
            for i in range(1, 81):
                for j in range(positions.shape[1]):
                    if positions[k][i][j][1]!=0 and positions[k][i][j][2]!=0 and probs[k][i][j]!=0:
                        print(label_2_name[i-1], " ", probs[k][i][j]*100,"%", "  ", positions[k][i][j][0], " ", positions[k][i][j][1], " ", positions[k][i][j][2], " ", positions[k][i][j][3])
    else:
        for j in range(positions.shape[1]):
            for i in range(1, 81):
                if positions[0][j][1]!=0 and positions[0][j][2]!=0 and probs[0][j][i]!=0:
                    print(label_2_name[i-1], " ", probs[0][j][i]*100,"%", "  ",positions[0][j][0], " ", positions[0][j][1], " ", positions[0][j][2], " ", positions[0][j][3])


@flow.function(func_config)
def yolo_user_op_eval_job():
    images, origin_image_info = yolo_predict_decoder(args.batch_size, args.image_height, args.image_width, args.image_list_path, "yolo")
    images = flow.identity(images, name="yolo-layer1-start")
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    yolo_pos_result = flow.identity(yolo_pos_result, name="yolo_pos_result_end")
    yolo_prob_result = flow.identity(yolo_prob_result, name="yolo_prob_result_end")
    return yolo_pos_result, yolo_prob_result

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

    def create_callback(step):
        def nop(ret):
            pass
        def callback(ret):
            yolo_pos, yolo_prob = ret
            print_detect_box(yolo_pos, yolo_prob)
            global cur_time
            if step==0:
                print("start_time:", time.time())
            elif step==args.total_batch_num-1:
                print("end time:", time.time())
            print(time.time()-cur_time)

            cur_time = time.time()

        if step % args.loss_print_steps == 0:
            return callback
        else:
            return nop



    for step in range(args.total_batch_num):
        yolo_user_op_eval_job().async_get(create_callback(step))
