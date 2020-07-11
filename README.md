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

## Inference

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

运行脚本后，将在data/result下生成检测后带bbox标记框的图片：

![detected_kite](data/result/detected_kite.jpg)







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



# Training

下面，将通过COCO数据集的训练过程为例，介绍YoloV3的训练过程，其他自定义数据集制作和格式，需保证和COCO类似。

## 1.准备数据集

### 数据集
这里使用COCO2014数据集，请提前准备/下载好COCO2014D训练集和验证集图片，将解压后的train2014和val2014放在data/COCO/images目录下（或ln创建软链接将images链接至数据集存放路径）

### 资源文件

准备资源文件：labels，5k.part，trainvalno5k.part

```
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
```

### 数据准备
在data/COCO目录下执行脚本：
```
# get label file
tar xzf labels.tgz

# set up image list
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

# copy label txt to image dir
find labels/train2014/ -name "*.txt"  | xargs -i cp {} images/train2014/
find labels/val2014/   -name "*.txt"  | xargs -i cp {} images/val2014/
```
执行脚本将自动解压缩labels.tgz文件，并在当前目录下生成5k.txt和trainvalno5k.txt，然后将labels/train2014和labels/val2014的的所有label txt文件复制到对应的训练集和验证集文件夹中(保证图片和label在同一目录)。



至此，完成整个数据集的准备过程。



## 2.训练
修改yolo_train.sh脚本中的参数，令：--image_path_file="data/COCO/trainvalno5k.txt"并执行：
```
sh yolo_train.sh
```
