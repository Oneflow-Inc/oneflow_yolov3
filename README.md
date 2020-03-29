# oneflow_yolov3
Yolov3 in OneFlow

## For development
1. clone this repo
2. in root directory of this project, run:
```
bash scripts/test.sh
```
## Run Yolov3 inference
1. download model to root directory of this project
链接:https://pan.baidu.com/s/16nSqnISE7U6vhcGAKaB0Rw  密码:1vhh
2. run
```
./yolo_predict.sh
```

## 使用自己的数据集预测
1. 准备图片，如data/images的示例
2. 将图片的路径写入一个文件中，如data/with_dir_data_names的示例
3. 修改yolo_predict.sh中的image_list_file为2中写入的图片路径列表文件
4. 运行./yolo_predict.sh