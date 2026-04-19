from ultralytics import YOLO
from predictor import Predictor

def main():
    # 模型加载
    model_path = r"runs\detect\runs\detect\train_20260419_220642\weights\best.pt"
    model = YOLO(model_path)
    
    # 初始化预测器
    predictor = Predictor(model=model, conf_threshold=0.3, iou_threshold=0.45, imgsz=1280)
    
    # 方式1：对文件夹内所有jpg图像进行预测
    results = predictor.predict_folder("pano_out", output_dir="runs/predicted")
    print(f"处理完成，共处理 {len(results)} 张图像")
    
    # 方式2：传入已读取的图像进行预测
    import cv2
    image = cv2.imread("pano_out/IMG_2374_pano_floodfill.jpg")
    if image is not None:
        preds = predictor.predict_from_image(image)
        print(f"检测到 {len(preds)} 个目标")
    for pred in preds:
        print(f"类别: {pred['class_name']}, 归一化坐标: {pred['bbox']}")

if __name__ == "__main__":
    main()