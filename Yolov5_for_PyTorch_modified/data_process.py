import torch
if torch.__version__ >= '1.8':
    import torch_npu
import os
import sys
from pathlib import Path
import moxing as mox
import argparse
    
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default="./Data/mnist.npz", help='path where the dataset is saved')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--nc', type=int, default=80, help='number of classes')
    parser.add_argument('--names', type=str, default='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,trafficlight,\
firehydrant,stopsign,parkingmeter,bench,bird,cat,dog,horse,sheep,cow,\
elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,\
skis,snowboard,sportsball,kite,baseballbat,baseballglove,skateboard,surfboard,\
tennisracket,bottle,wineglass,cup,fork,knife,spoon,bowl,banana,apple,\
sandwich,orange,broccoli,carrot,hotdog,pizza,donut,cake,chair,couch,\
pottedplant,bed,diningtable,toilet,tv,laptop,mouse,remote,keyboard,cellphone,\
microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddybear,\
hairdrier,toothbrush', help='class names')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    
    parentPath = str(FILE.parents[1])
    localPath = str(FILE.parents[0])
    coco_file_path = localPath + '/coco'
    
    mox.file.copy_parallel(coco_file_path,opt.data_url)
    os.chdir(opt.data_url)
    os.system("pwd")
    
    namesList = opt.names.split(',')
    
    if opt.nc != len(namesList):
        assert "nc need same as the number of names"
    
    with open('coco_class.txt','w') as f:
        for index in range(opt.nc):
            strLine = str(index) + "," + namesList[index] + "\n"
            print('strLine:',strLine)
            f.write(strLine)
            
    #基于coco2017格式数据集生成yolov5专用标注文件
    os.system("python coco2yolo.py")
            
    os.chdir(parentPath)
    os.system("pwd")
    
    data_file_path = localPath + '/data/coco.yaml'
   
    #更新coco.yaml配置文件的数据集类别数目和名称
    with open(data_file_path,'w') as f:
        f.write('path: ' + opt.data_url + '\n')
        f.write('train: train2017.txt\n')
        f.write('val: val2017.txt\n')
        f.write('test: test-dev2017.txt\n')
        
        f.write('nc: ' + str(opt.nc) + '\n')   
        namesFile = ""
        for index in range(opt.nc):
            namesFile = namesFile + "'" + namesList[index] + "'"
            if index + 1 != opt.nc:
                namesFile = namesFile + ","
        f.write('names: [' + namesFile + ']\n')