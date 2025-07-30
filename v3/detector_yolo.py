import torch
from ultralytics import YOLO
from PIL import Image

# ======================
# 加载模型
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/detect/yolo_finetune_coco/weights/best.pt"  # 你的YOLO微调权重路径
CONF_THRESHOLD = 0.25  # 置信度阈值

model = YOLO(MODEL_PATH)
model.to(device)
model.eval()

# ======================
# 检测函数
# ======================
def detect_boxes(image_path: str, label: str):
    """
    使用微调后的YOLO模型检测图像中的目标，仅返回指定类别的bbox列表。
    返回: [bbox[xmin,ymin,xmax,ymax], ...]
    """
    # 执行预测
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False
    )

    bboxes = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # [xmin, ymin, xmax, ymax]
        scores = r.boxes.conf.cpu().numpy()  # 置信度
        labels = r.boxes.cls.cpu().numpy()   # 类别索引

        for box, score, lbl_idx in zip(boxes, scores, labels):
            lbl_name = model.names[int(lbl_idx)]
            if lbl_name.lower() == label.lower():  # 仅保留匹配标签
                bboxes.append(box.tolist())

    return bboxes

# ======================
# 示例测试
# ======================
if __name__ == "__main__":
    img_path = "datasets_yolo/images/val/0.jpg"
    target_label = "vehicle"
    boxes = detect_boxes(img_path, target_label)
    print(f"检测到的 {target_label} 边界框数量: {len(boxes)}")
    for b in boxes:
        print(b)
