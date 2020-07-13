model_dir=of_model/yolov3_model_python/
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/python_load_data_predict_with_print_box.py \
--gpu_num_per_node=1 --batch_size=4  \
--model_load_dir=$model_dir \
--label_path=data/coco.names  --use_tensorrt=1 \
--input_dir=data/images \
--output_dir=data/result \
# --image_paths 'data/images/000002.jpg' 'data/images/000004.jpg' 'data/images/kite.jpg'