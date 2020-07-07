model_dir=of_model/yolov3_model_python/
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/predict_with_print_box.py \
--model_load_dir=$model_dir \
--label_path=data/coco.names \
--input_dir=data/images \
--output_dir=data/result
# --image_paths 'data/images/000002.jpg' 'data/images/000004.jpg' 'data/images/kite.jpg'
