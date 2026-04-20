import cv2
import os
from ultralytics.models.sam import SAM3SemanticPredictor

# 1. 从视频中截取最中间的一帧
def extract_frame(video_path, output_path):
    """从视频中提取指定帧"""

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames//2)  # 帧号从0开始
    
    # 读取帧
    ret, frame = cap.read()
    
    if ret:
        # 保存帧为图像
        cv2.imwrite(output_path, frame)
    
    # 释放视频捕获对象
    cap.release()
    return ret

# 2. 初始化预测器（配置参数可按需调整）
overrides = dict(
    conf=0.4,  # 置信度阈值，默认0.25
    task="segment",  # 任务类型：分割
    mode="predict",  # 运行模式：预测
    model="sam3.pt",  # 模型权重路径（相对/绝对路径）
    half=True,  # 启用FP16精度，提升推理速度（需GPU支持）
    save=True,
    #save_dir="./mask_folder",
    show_labels=False,  # 禁用类别标签显示
    show_conf=True,  # 启用置信度显示
)
predictor = SAM3SemanticPredictor(
    overrides=overrides
)

# 3. 提取视频帧
video_name = "IMG_2376"
video_path = f"vedio_exp/{video_name}.MOV"  # 替换为实际视频路径
frame_output_path = f"results/{video_name}.png" # 最中间帧图片的保存路径

extract_frame(video_path, frame_output_path)

# 4. 设置待分割图像
predictor.set_image(frame_output_path)
        
# 5. 执行分割
results = predictor(text=["an entire screw"], save=False, show_labels=False)  # 禁用自动保存，以便手动控制

# 手动保存结果，使用自定义文件名
if results:
    mask_output_path = f"./mask_folder/{video_name}_mask.png"
    # 确保保存目录存在
    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    # 保存分割结果，使用正确的参数名称
    results[0].save(mask_output_path, labels=False, conf=True)
    #print(f"已保存分割结果到: {mask_output_path}")

       
    