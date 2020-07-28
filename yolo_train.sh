model_dir=of_model/of_yolov3_model/
rm -r save_model
mkdir save_model
# export ONEFLOW_DEBUG_MODE=""

# gdb --args
python3 oneflow_yolov3/model/yolo_train.py \
--gpu_num_per_node=1 --batch_size=4 --base_lr=0.001 \
--num_epoch=130 --model_load_dir=$model_dir \
--classes=80 --num_boxes=90 --save_frequency=1 \
--model_save_dir="save_model" \
--dataset_dir="data/test_trainvalno5k.txt"
# --dataset_dir="data/COCO/trainvalno5k.txt"
