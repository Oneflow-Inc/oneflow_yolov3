# Set up model dir,e.g.of_yolov3_model(mAP: 56.5%);
model_dir=of_model/of_yolov3_model/
export ONEFLOW_DEBUG_MODE=""

# gdb --args \
python3 oneflow_yolov3/model/yolo_evaluate.py \
--gpu_num_per_node=1 --batch_size=1  \
--model_load_dir=$model_dir \
--label_path=data/coco.names  --use_tensorrt=1 \
--nms_thres=0.5 \
--conf_thres=0.001 \
--iou_thres=0.5 \
--image_paths=data/COCO/5k.txt