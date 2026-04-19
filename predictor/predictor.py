"""
predictor.py - 预测接口
"""

from pathlib import Path
from ultralytics import YOLO


class Predictor:
    """预测器类"""
    
    def __init__(self, model, conf_threshold=0.25, iou_threshold=0.45, imgsz=1280):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        self.class_names = {
            0: 'Type1',
            1: 'Type2',
            2: 'Type3',
            3: 'Type4',
            4: 'Type5'
        }
    
    def predict_folder(self, folder_path, output_dir="runs/predicted"):
        """对文件夹内所有jpg图像进行预测"""
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(folder_path.glob("*.jpg"))
        results_dict = {}
        
        for img_path in images:
            result = self.model(img_path, 
                               conf=self.conf_threshold, 
                               iou=self.iou_threshold,
                               agnostic_nms=True,
                               verbose=False,
                               imgsz=self.imgsz,
                               rect=True,)
            
            img_h, img_w = result[0].orig_shape
            boxes = result[0].boxes
            
            if len(boxes) == 0:
                results_dict[img_path.name] = []
                label_path = output_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    pass
            else:
                preds = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1_norm = x1 / img_w
                    y1_norm = y1 / img_h
                    x2_norm = x2 / img_w
                    y2_norm = y2 / img_h
                    
                    preds.append({
                        'class_id': int(box.cls[0]),
                        'class_name': self.class_names[int(box.cls[0])],
                        'bbox': [x1_norm, y1_norm, x2_norm, y2_norm]
                    })
                results_dict[img_path.name] = preds
                
                label_path = output_dir / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    for pred in preds:
                        x1, y1, x2, y2 = pred['bbox']
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        f.write(f"{pred['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return results_dict
    
    def predict_from_image(self, image):
        """传入已读取的图像进行预测"""
        results = self.model(image, 
                            conf=self.conf_threshold, 
                            iou=self.iou_threshold,
                            agnostic_nms=True,
                            verbose=False,
                            imgsz=self.imgsz,
                            rect=True)
        
        img_h, img_w = results[0].orig_shape
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return []
        
        preds = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_norm = x1 / img_w
            y1_norm = y1 / img_h
            x2_norm = x2 / img_w
            y2_norm = y2 / img_h
            
            preds.append({
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'bbox': [x1_norm, y1_norm, x2_norm, y2_norm]
            })
        
        return preds