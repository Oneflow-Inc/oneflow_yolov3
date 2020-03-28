# oneflow_yolov3
Yolov3 in OneFlow

## For development
1. clone this repo
2. in root directory of this project, run:
```
bash scripts/test.sh
```
## Run Yolov3 inference
```
./yolo_predict.sh
```
## use Tensorrt
1. download TensorRT-6.0.1.5
2. build oneflow WITH_TENSORRT
```
cmake -DWITH_TENSORRT=ON -DTENSORRT_ROOT=$tensorrt_dir -DTHIRD_PARTY=ON ..
```
