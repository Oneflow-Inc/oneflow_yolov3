model_dir=of_model/yolov3_model_python/
# image_list_file=data/with_dir_data_names
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/predict_with_print_box.py \
--gpu_num_per_node=1 --batch_size=1  \
--model_load_dir=$model_dir \
--label_path=data/coco.names --use_tensorrt=0  \
--image_path_list 'data/images/000001.jpg' 'data/images/000002.jpg' 'data/images/000003.jpg' 'data/images/000004.jpg' 'data/images/000011.jpg' 'data/images/000012.jpg' 'data/images/000013.jpg' 'data/images/000014.jpg' 'data/images/kite.jpg'
# --image_path_list 'data/images/000001.jpg'
