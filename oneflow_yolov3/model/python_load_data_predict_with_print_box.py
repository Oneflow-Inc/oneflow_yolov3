import oneflow as flow
import time
import argparse
import os
from yolo_net import YoloPredictNet
from data_preprocess import image_preprocess_v2, batch_image_preprocess_v2
import oneflow_yolov3
import utils
import math

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-load", "--pretrained_model", type=str, required=True)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-label_path", "--label_path", type=str, required=True)
parser.add_argument("-batch_size", "--batch_size", type=int, default=8, required=False)
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
#func_config.tensorrt.use_fp16()
nms = True


input_blob_def_dict = {
    "images": flow.FixedTensorDef((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float),
    "origin_image_info": flow.FixedTensorDef((args.batch_size, 2), dtype=flow.int32),
}


@flow.global_function(func_config)
def yolo_user_op_eval_job(images=input_blob_def_dict["images"], origin_image_info=input_blob_def_dict["origin_image_info"]):
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    return yolo_pos_result, yolo_prob_result, origin_image_info


if __name__ == "__main__":
    assert os.path.exists(args.pretrained_model)
    assert args.input_dir or os.path.exists(args.image_paths[0])
    assert os.path.exists(args.label_path)

    if args.input_dir and os.path.exists(args.input_dir):
        args.image_paths = [args.input_dir + os.sep + path for path in os.listdir(args.input_dir)]

    flow.config.gpu_device_num(args.gpu_num_per_node)
    # load model
    check_point = flow.train.CheckPoint()
    check_point.load(args.pretrained_model)

    # Note: if use python_nms, than yolo_net.py should be nms=False
    python_nms = True

    path_list = args.image_paths
    iter_num = math.floor(len(args.image_paths)/float(args.batch_size))
    for i in range(iter_num):
        paths = path_list[i*args.batch_size:(i+1)*args.batch_size]
        images, origin_image_info = batch_image_preprocess_v2(paths, args.image_height, args.image_width)
        start = time.time()
        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(images, origin_image_info).get()
        print('cost: %.4f ms' % (1000 * (time.time() - start)))

        if python_nms:
            bboxes = utils.batch_postprocess_boxes_new(yolo_pos, yolo_prob, origin_image_info, 0.3)
        else:
            bboxes = utils.batch_postprocess_boxes(yolo_pos, yolo_prob, origin_image_info, 0.3)

        utils.save_detected_images(paths, bboxes, args.label_path, args.output_dir)
        print('iter:%d >> bboxes:' % i, bboxes,
              '\n------------------------------------------------------------------------')


