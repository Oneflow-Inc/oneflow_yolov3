import oneflow as flow
import time
import argparse
import os
from yolo_net import YoloPredictNet
import oneflow_yolov3
from oneflow_yolov3.ops.yolo_decode import yolo_predict_decoder
import utils
import numpy as np

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, required=True)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-label_path", "--label_path", type=str, required=True)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-use_tensorrt", "--use_tensorrt", type=int, default=0, required=False)
parser.add_argument("-input_dir", "--input_dir", type=str, required=False)
parser.add_argument("-output_dir", "--output_dir", type=str, default='data/result', required=False)
parser.add_argument("-image_paths", "--image_paths", type=str, nargs='+', required=False)

args = parser.parse_args()


flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)
# func_config.tensorrt.use_fp16()
nms = True

def test_process(yolo_pos, yolo_prob, valid_num, origin_image_info):
    for i in range(args.batch_size):
        boxes = []
        probs = []
        for k in range(len(yolo_pos)): # 3
            layer_boxes = yolo_pos[k].ndarray()
            layer_probs = yolo_prob[k].ndarray()
            layer_valid_num = valid_num[k].ndarray()
            print(layer_valid_num)
            boxes.append(layer_boxes[i, 0:layer_valid_num[i]])
            probs.append(layer_probs[i, 0:layer_valid_num[i]])
        boxes = np.concatenate(boxes)
        probs = np.concatenate(probs)
        print("batch_idx", i)
        print("boxes", boxes.shape, boxes)
        print("probs", probs.shape, probs[:,0:5])

@flow.global_function(func_config)
def yolo_user_op_eval_job():
    images, origin_image_info = yolo_predict_decoder(args.batch_size, args.image_height,
                                                     args.image_width, args.image_paths, "yolo")
    yolo_pos_result, yolo_prob_result, valid_num_result = YoloPredictNet(images, origin_image_info, trainable=False)
    return yolo_pos_result, yolo_prob_result, valid_num_result, origin_image_info


if __name__ == "__main__":
    assert os.path.exists(args.model_load_dir)
    assert args.input_dir or os.path.exists(args.image_paths[0])
    assert os.path.exists(args.label_path)

    if args.input_dir and os.path.exists(args.input_dir):
        args.image_paths = [args.input_dir + os.sep + path for path in os.listdir(args.input_dir)]

    flow.config.gpu_device_num(args.gpu_num_per_node)  # set gpu num
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    for i in range(1):
        start = time.time()
        yolo_pos, yolo_prob, valid_num, origin_image_info = yolo_user_op_eval_job().get()
        test_process(yolo_pos, yolo_prob, valid_num, origin_image_info)
        #print('cost: %.4f ms' % (1000 * (time.time() - start)))
        #bboxes = utils.postprocess_boxes(yolo_pos[0], yolo_prob[0], origin_image_info[0], 0.3)
        #utils.save_detected_images([args.image_paths[i]], [bboxes], args.label_path, args.output_dir)
        #print('%s >> bboxes:' % args.image_paths[i], bboxes,
        #      '\n------------------------------------------------------------------------')

