import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from typing import List
from config import GDINO_BOX_THRESHOLD

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID  = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()


def detect_boxes(image_path: str, phrase: str) -> List[List[int]]:
    """
    调用 Grounding-DINO 做零样本检测，返回 [[xmin,ymin,xmax,ymax], …]。
    出错或无命中返回 []。
    """
    try:
        image = Image.open(image_path).convert("RGB")

        text = phrase.strip().lower()
        if not text.endswith("."):
            text += "."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=0.0,
            target_sizes=[image.size[::-1]],   # (H,W)
        )

        boxes = results[0]["boxes"].cpu().numpy().tolist()
        return boxes
    except Exception:
        # 任意异常均视为“没有检测到”
        return []
