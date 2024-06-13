#!/bin/bash

# MA preset envs
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
# also indicates NPU per node
NGPUS_PER_NODE="$MA_NUM_GPUS"

# self-define, it can be changed to >=10000 port
MASTER_PORT="38888"

model_name=s
batch_size=512
epochs=300
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
python ${MA_JOB_DIR}/Yolov5_for_PyTorch/data_process.py --data_url $data_url --nc $nc --names $names

# replace ${MA_JOB_DIR}/code/torch_ddp.py to the actutal training script
PYTHON_SCRIPT=${MA_JOB_DIR}/Yolov5_for_PyTorch/train.py
PYTHON_ARGS="--batch-size $batch_size --epochs $epochs --data_url $data_url --train_url $train_url --model_name $model_name"

export HCCL_WHITELIST_DISABLE=1

# set npu plog env
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${task_plog_path}

echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

# set hccl timeout time in seconds
export HCCL_CONNECT_TIMEOUT=1800

#8卡训练
# replace ${ANACONDA_DIR}/envs/${ENV_NAME}/bin/python to the actual python
CMD="${ANACONDA_DIR}/envs/${ENV_NAME}/bin/python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NGPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --use_env \
    $PYTHON_SCRIPT \
    $PYTHON_ARGS
"
echo $CMD
$CMD

#单卡验证
python ${MA_JOB_DIR}/Yolov5_for_PyTorch/val.py --img-size 640 --weight 'yolov5.pt' --batch-size 128 --device 0 --half

#将模型进行转换，以便在GPU环境可以推理，并输出到指定路径
python ${MA_JOB_DIR}/Yolov5_for_PyTorch/export_ma.py --train_url $train_url

