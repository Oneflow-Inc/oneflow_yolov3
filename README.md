# oneflow_yolov3
Yolov3 in OneFlow

## For development
1. clone this repo

2. 需确认在当前为python3环境，且可以`import oneflow`成功

3. 安装python依赖库
```shell
   pip install -r requirements.txt
```
4. 在项目root目录下，执行:
```
bash scripts/test.sh
```
执行此脚本，将cpp代码中自定义的op算子编译成可调用执行的.so文件，您将在项目路径下看到：libdarknet.so和liboneflow_yolov3.so

## Run Yolov3 inference

1. 下载模型到项目root目录下，解压并放置文件夹of_model于项目root目录下。
链接:https://pan.baidu.com/s/16nSqnISE7U6vhcGAKaB0Rw  密码:1vhh
2. 运行
```
./yolo_predict.sh
```
或
```
./yolo_predict_python_data_preprocess.sh
```

## 支持的数据格式
1. 如果网络初始化前可以确定要做推理的图片
- 可以将文件路径写入一个文件filename中，配置image_list_path参数执行
- 也可以直接传图片路径列表，配置image_path_list参数执行
可参考yolo_predict.sh
2. 如果网络初始化前不能确定推理的图片
- 每次执行时传入图片路径执行
可参考yolo_predict_python_data_preprocess.sh


## 使用自己的数据集预测
1. 准备图片，如data/images的示例
2. 将图片的路径写入一个文件中，如data/with_dir_data_names的示例
3. 修改yolo_predict.sh中的image_list_file为2中写入的图片路径列表文件
4. 运行./yolo_predict.sh


## 说明
目前如果调用yolo_predict.sh执行，数据预处理部分对darknet有依赖。
，其中：predict decoder中调用load_image_color、letterbox_image函数  
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



# 用COCO数据集训练

## 1.准备数据集
### 数据集
数据集主要包含训练集和验证集图片，将解压后的train2014和val2014放在data/COCO/images目录下（或ln创建软链接将images链接至数据集存放路径）

### 资源文件
```
wget -c https://pjreddie.com/media/files/coco/labels.tgz
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
```

### 数据准备
在data/COCO目录下执行脚本：
```
sh get_coco_dataset.sh
```
执行脚本将自动解压缩labels.tgz文件，并在当前目录下生成5k.txt和trainvalno5k.txt

## 训练
修改yolo_train.sh脚本中的参数，令：--image_path_file="data/COCO/trainvalno5k.txt"并执行：
```
sh yolo_train.sh
```
