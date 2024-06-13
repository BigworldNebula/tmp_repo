# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

class Yolov5():
    def __init__(self, weights, conf_thres=0.25, iou_thres=0.45, device='cpu'):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.weights = weights
        self.imgsz=[640,640]  # inference size (pixels)
        self.conf_thres=conf_thres  # confidence threshold
        self.iou_thres=iou_thres  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.save_txt=False  # save results to *.txt
        self.save_conf=False,  # save confidences in --save-txt labels
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.line_thickness=3  # bounding box thickness (pixels)
        self.half=False  # use FP16 half-precision inference
        
        self.save_img = True  # save inference images
        

        # Initialize
        set_logging()
        self.device = select_device(device)
        
        if self.half and self.device.type != 'cpu':
            self.half = True
        else:
            self.half = False
        
        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=self.device)
        
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
    
    @torch.no_grad()
    def inference(self,source):
        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=False)  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=False)  # make dir
        
        # Dataloader
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=True)
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        dt, seen = [0.0, 0.0, 0.0], 0

        pred_result = []

        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(img, augment=False, visualize=False)[0]
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image

                seen += 1
                p, s, im0 = path, '', im0s.copy()

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem)  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{self.names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    for x1, y1, x2, y2, conf, cls in reversed(det):
                        c = int(cls)
                        det_result = self.names[c], c, float(x1), float(y1), float(x2), float(y2), float(conf)
                        pred_result.append(det_result)


                # Print time (inference-only)
                print(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                im0 = annotator.result()

                # Save results (image with detections)
                if self.save_img:
                    cv2.imwrite(save_path, im0)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")  
            
        return pred_result
