import torch
if torch.__version__ >= '1.8':
    import torch_npu
import os
import sys
import moxing as mox
import argparse
from pathlib import Path
    
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_url', type=str, default="./Model", help='path where the model is saved')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    
    #将NPU训练生成的模型通过CPU方式加载保存，从而可以在GPU环境中推理
    tmp = torch.load('yolov5.pt',map_location='cpu')
    torch.save(tmp, 'new_yolov5.pt')
    
    #将生成的模型和训练文件拷贝到输出路径中
    copy_path = str(FILE.parents[0]) + "/runs/train/exp"
    mox.file.copy_parallel(copy_path,opt.train_url)
    infer_path = str(FILE.parents[0]) + "/infer"
    mox.file.copy_parallel(infer_path,opt.train_url)
    mox.file.copy("new_yolov5.pt",os.path.join(opt.train_url,"yolov5.pt"))