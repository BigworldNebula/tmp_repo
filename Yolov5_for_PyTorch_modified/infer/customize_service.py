import os
import glob
import shutil
from detect_ma import Yolov5
from model_service.pytorch_model_service import PTServingBaseService

class yolov5_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:',model_name)
        print('model_path:',model_path)
                
        self.model = Yolov5(model_path,conf_thres=0.25, iou_thres=0.45, device='cpu')
        
        self.capture = "test.png"

    def _preprocess(self, data):
        # preprocessed_data = {}
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"
    
    def _inference(self, data):
        pred_result = self.model.inference(self.capture)
        return pred_result

    def _postprocess(self, data):
        result = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []
        
        for pred in data:
            classes, _, x1, y1, x2, y2, conf = pred
            detection_classes.append(classes)
            boxes = [y1,x1,y2,x2]
            detection_boxes.append(boxes)
            detection_scores.append(conf)
                
        result['detection_classes'] = detection_classes
        result['detection_boxes'] = detection_boxes
        result['detection_scores'] = detection_scores
            
        print('result:',result)    
        return result
