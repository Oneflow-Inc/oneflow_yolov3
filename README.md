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


## 说明
目前数据预处理部分对darknet有依赖，其中：  
predict decoder中调用load_image_color、letterbox_image函数  
train decoder中调用load_data_detection函数  
主要涉及以下操作，在后续的版本中会使用oneflow decoder ops替换  
1. image read  
2. nhwc -> nchw  
3. image / 255  
4. bgr2rgb  
5. resize_image  
6. fill_image   
7. random_distort_image  
8. clip image  
9. random flip image and box  
10. randomize_boxes   
11. correct_boxes  