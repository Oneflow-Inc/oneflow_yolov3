# Set up model dir,e.g.of_yolov3_model; yolov3_snapshot_50
model_dir=of_model/of_yolov3_model/
export ONEFLOW_DEBUG_MODE=""

python3 oneflow_yolov3/model/predict_with_print_box.py \
--pretrained_model=$model_dir \
--label_path=data/coco.names \
--input_dir=data/images \
--output_dir=data/result
# --image_paths 'data/images/000002.jpg' 'data/images/000004.jpg' 'data/images/kite.jpg'
