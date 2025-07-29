import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config import GDINO_BOX_THRESHOLD
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-base"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()

def detect_boxes(image_path: str, phrase: str):
    image = Image.open(image_path).convert("RGB")

    text = phrase.lower().strip()
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
        target_sizes=[image.size[::-1]],
    )

    detections = results[0]
    boxes = detections["boxes"].cpu().numpy().tolist()
    scores = detections["scores"].cpu().numpy().tolist()

    # 返回 bbox+score
    return [(b, s) for b, s in zip(boxes, scores) if s >= GDINO_BOX_THRESHOLD]

