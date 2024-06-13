#!/bin/bash

model_name=s
batch_size=48
epochs=1
data_url=dataset
train_url=model
nc=80
names="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,trafficlight,\
firehydrant,stopsign,parkingmeter,bench,bird,cat,dog,horse,sheep,cow,\
elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,\
skis,snowboard,sportsball,kite,baseballbat,baseballglove,skateboard,surfboard,\
tennisracket,bottle,wineglass,cup,fork,knife,spoon,bowl,banana,apple,\
sandwich,orange,broccoli,carrot,hotdog,pizza,donut,cake,chair,couch,\
pottedplant,bed,diningtable,toilet,tv,laptop,mouse,remote,keyboard,cellphone,\
microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddybear,\
hairdrier,toothbrush"

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   elif [[ $para == --epochs* ]];then
      	epochs=`echo ${para#*=}`
   elif [[ $para == --data_url* ]];then
      	data_url=`echo ${para#*=}`
   elif [[ $para == --train_url* ]];then
      	train_url=`echo ${para#*=}`
   elif [[ $para == --nc* ]];then
      	nc=`echo ${para#*=}`
   elif [[ $para == --names* ]];then
      	names=`echo ${para#*=}`
   fi
done

#数据集处理，转成专用格式，并修改配置文件
python Yolov5_for_PyTorch/data_process.py --data_url $data_url --nc $nc --names $names

#训练
python3 Yolov5_for_PyTorch/train.py --batch-size $batch_size --device 0 --epochs $epochs --imgsz 640 --freeze 0 --data_url $data_url --train_url $train_url --model_name $model_name --workers 8

#验证
python Yolov5_for_PyTorch/val.py --img-size 640 --weight 'yolov5.pt' --batch-size 128 --device 0 --half

#将模型进行转换，以便在GPU环境可以推理，并输出到指定路径
python Yolov5_for_PyTorch/export_ma.py --train_url $train_url

