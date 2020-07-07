model_dir=of_model/yolov3_model_python/
rm -r save_model
mkdir save_model
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/yolo_train.py --gpu_num_per_node=1 --batch_size=4 --base_lr=0.001 --image_path_file="test_trainvalno5k.txt" --total_batch_num=100 --model_load_dir=$model_dir --classes=80 --num_boxes=90 --num_of_batches_in_snapshot=100 --model_save_dir="save_model"

