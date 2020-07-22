# model_dir=yolov3_pretrained
model_dir=of_model/yolov3_model_python/  # yolov3_snapshot_50
export ONEFLOW_DEBUG_MODE=""

# gdb --args \
python3 oneflow_yolov3/model/yolo_test_python.py \
--gpu_num_per_node=1 --batch_size=1  \
--model_load_dir=$model_dir \
--label_path=data/coco.names  --use_tensorrt=1 \
--nms_thres=0.5 \
--conf_thres=0.001 \
--iou_thres=0.5 \
--image_paths=data/COCO/5k.txt  # 5k_2017.txt