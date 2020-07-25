model_dir=of_model/of_yolov3_model/
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/python_load_data_predict_with_print_box.py \
--gpu_num_per_node=1 --batch_size=4  \
--pretrained_model=$model_dir \
--label_path=data/coco.names  --use_tensorrt=1 \
--input_dir=data/images \
--output_dir=data/result \
# --image_paths 'data/images/000002.jpg' 'data/images/000004.jpg' 'data/images/kite.jpg'