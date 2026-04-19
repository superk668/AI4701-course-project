支持两种预测方式：
1. 对文件夹内所有图像进行批量预测,需要将上一步拼接的图像保存在输入文件夹中
2. 对单张已读取的图像进行预测，可以直接传入上一步拼接的图像，节省保存再读取的时间

###在主函数中如何使用
from ultralytics import YOLO
from pridictor import Pridictor

# 加载模型
model = YOLO("model_path/best.pt")

# 初始化预测器
predictor = Predictor(
    model=model,              # 已加载的YOLO模型
    conf_threshold=0.3,      # 置信度阈值（可选，默认0.25）
    iou_threshold=0.45,      # NMS的IoU阈值（可选，默认0.45）
    imgsz=640                # 推理图像尺寸（可选，默认640）
)


## 接口1：predict_folder

对文件夹内所有图像进行批量预测，结果保存在指定目录。

### 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `folder_path` | str | 是 | 图像文件夹路径 |
| `output_dir` | str | 否 | 输出目录，默认为 `"runs/predicted"` |

### 输出格式

**返回值**：`dict`

```python
{
    "图像名1.jpg": [
        {
            "class_id": 0,           # 类别ID（int）
            "class_name": "Type1",   # 类别名称（str）
            "bbox": [x1, y1, x2, y2] # 边界框坐标（list of float）
        },
        ...
    ],
    "图像名2.jpg": [...]
}
```

**保存文件**：在 `output_dir` 目录下，每张图像生成一个同名的 `.txt` 文件,坐标是yolo格式的归一化坐标

标签文件格式（每行）：
```
class_id x1 y1 x2 y2
```

示例：
```
2 0.383983 0.278989 0.022414 0.056435
0 0.392589 0.514717 0.020046 0.041137
0 0.372316 0.465781 0.018637 0.037868
```

### 使用示例

```python
# 对 pano_out 文件夹内所有图像进行预测
results = predictor.predict_folder("pano_out", output_dir="runs/predicted")

# 遍历结果
for img_name, preds in results.items():
    print(f"{img_name}: {len(preds)} 个目标")
    for pred in preds:
        print(f"  类别: {pred['class_name']}, 边界框: {pred['bbox']}")
```

## 接口2：predict_from_image

对已读取的单张图像进行预测。

### 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | numpy.ndarray | 是 | BGR格式的图像数组（如cv2.imread读取的结果） |

### 输出格式

**返回值**：`list`

```python
[
    {
        "class_id": 0,           # 类别ID（int）
        "class_name": "Type1",   # 类别名称（str）
        "bbox": [x1, y1, x2, y2] # 边界框坐标（list of float）
    },
    ...
]
```

如果没有检测到目标，返回空列表 `[]`。

### 使用示例

# 读取图像
image = cv2.imread("pano_out/00000.jpg")

# 预测
preds = predictor.predict_from_image(image)

# 处理结果
print(f"检测到 {len(preds)} 个目标")
for pred in preds:
    print(f"类别: {pred['class_name']}, 边界框: {pred['bbox']}")
```

### 输出示例
类别: Type3, 归一化坐标: [0.37277636920588614, 0.25077080160884535, 0.3951900848798657, 0.3072062088271319]
类别: Type1, 归一化坐标: [0.3825661307402902, 0.49414874577926376, 0.4026122374079616, 0.5352853548728813]
类别: Type1, 归一化坐标: [0.36299728829820443, 0.4468468617584746, 0.3816339306859283, 0.4847148313360699]
类别: Type4, 归一化坐标: [0.32521253185686144, 0.5480767201569121, 0.3538825507825956, 0.593881484209481]

## 类别映射

| 类别ID | 类别名称 |
|--------|----------|
| 0 | Type1 |
| 1 | Type2 |
| 2 | Type3 |
| 3 | Type4 |
| 4 | Type5 |
