model_dir=of_model/yolov3_model_python/
image_list_file=data/with_dir_data_names
label_2_name_file=data/coco.names
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/predict_with_print_box.py --total_batch_num=9 \
--gpu_num_per_node=1 --batch_size=1  --model_load_dir=$model_dir \
--image_list_path=$image_list_file --label_to_name_file=$label_2_name_file \
--use_tensorrt=0  \
--image_path_list 'data/images/000001.jpg' 'data/images/000002.jpg' 'data/images/000003.jpg' 'data/images/000004.jpg' 'data/images/000011.jpg' 'data/images/000012.jpg' 'data/images/000013.jpg' 'data/images/000014.jpg'

