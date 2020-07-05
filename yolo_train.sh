model_dir=of_model/yolov3_model_python/
image_path_file=data/with_dir_data_names
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/yolo_train.py --gpu_num_per_node=1 --batch_size=32  --image_path_file="gr_trainvalno5k.txt" --model_load_dir=$model_dir 

