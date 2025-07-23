# detector.py

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config import GDINO_BOX_THRESHOLD
from PIL import Image

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 固定模型 ID
MODEL_ID = "IDEA-Research/grounding-dino-base"

# 全局加载 Processor 和 Model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()


def detect_boxes(image_path: str, phrase: str):
    """
    对给定图像与单个文本短语做零样本检测，返回
    List[List[xmin, ymin, xmax, ymax]]，按 GDINO_BOX_THRESHOLD 过滤。
    """
    # 1. 读取图像
    image = Image.open(image_path).convert("RGB")

    # 2. 构造文本查询：小写并以句号结尾（HF 接口要求）
    text = phrase.lower().strip()
    if not text.endswith("."):
        text += "."

    # 3. 预处理输入
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    ).to(device)

    # 4. 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. 后处理：得到一个长度为 batch_size（这里=1）的 list
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=GDINO_BOX_THRESHOLD,
        text_threshold=0.0,
        target_sizes=[image.size[::-1]],  # PIL size is (W, H) -> target_sizes expects (H, W)
    )

    # 6. 提取 boxes
    # results[0] 是 dict，包含 "boxes" (Tensor[N,4]), "scores", "labels"
    detections = results[0]
    boxes = detections["boxes"].cpu().numpy().tolist()

    return boxes
