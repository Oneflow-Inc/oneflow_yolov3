import oneflow as flow
import time
import argparse
import os
from yolo_net import YoloPredictNet
from data_preprocess import image_preprocess_v2
import oneflow_yolov3
import utils

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, required=True)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-label_path", "--label_path", type=str, required=True)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-use_tensorrt", "--use_tensorrt", type=int, default=0, required=False)
parser.add_argument("-image_path_list", "--image_path_list", type=str, nargs='+', required=True)


args = parser.parse_args()


flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)
#func_config.tensorrt.use_fp16()
nms = True


input_blob_def_dict = {
    "images": flow.FixedTensorDef((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float),
    "origin_image_info": flow.FixedTensorDef((args.batch_size, 2), dtype=flow.int32),
}


@flow.global_function(func_config)
def yolo_user_op_eval_job(images=input_blob_def_dict["images"], origin_image_info=input_blob_def_dict["origin_image_info"]):
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    yolo_pos_result = flow.identity(yolo_pos_result, name="yolo_pos_result_end")
    yolo_prob_result = flow.identity(yolo_prob_result, name="yolo_prob_result_end")
    return yolo_pos_result, yolo_prob_result, origin_image_info


if __name__ == "__main__":
    assert os.path.exists(args.model_load_dir)
    assert os.path.exists(args.image_path_list[0])
    assert os.path.exists(args.label_path)

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.ctrl_port(9789)
    # load model
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    image_list = args.image_path_list
    coco_label_path = args.label_path

    for i in range(len(image_list)):
        images, origin_image_info = image_preprocess_v2(image_list[i], args.image_height, args.image_width)
        start = time.time()
        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(images, origin_image_info).get()
        end = time.time()
        bboxes = utils.postprocess_boxes(yolo_pos, yolo_prob, origin_image_info[0], 0.3)
        utils.save_detected_result(image_list[i], bboxes, coco_label_path)
        print('%s >>> bboxes:' % image_list[i], bboxes)
        print('cost time: %.4f ms\n--------------------------------------------------------------'
              % (1000 * (end - start)))
