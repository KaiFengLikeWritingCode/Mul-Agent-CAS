import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# =====================
# 参数配置
# =====================
MODEL_ID = "IDEA-Research/grounding-dino-base"
DATASET_JSON = "./coco_annotations2.json"
IMAGE_DIR = "./datasets_0/sample_image"
OUTPUT_DIR = "./grounding_dino_finetuned"
EPOCHS = 10
BATCH_SIZE = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# 加载模型与处理器
# =====================
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.train()


# =====================
# 数据集定义
# =====================
class CocoDetectionDataset(Dataset):
    def __init__(self, json_file, image_dir, processor):
        with open(json_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # 将 annotations 按 image_id 分组
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann["image_id"], []).append(ann)

        # 只保留有标注的图片
        self.ids = [img_id for img_id, anns in self.image_to_anns.items() if len(anns) > 0]
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns[image_id]
        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])  # 转换为 (x_min,y_min,x_max,y_max)
            labels.append(self.categories[ann["category_id"]])

        encoding = processor(images=image, text=labels, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": labels
        }


# =====================
# collate_fn: 处理变长标注
# =====================
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    bboxes = [item["bboxes"] for item in batch]  # list of tensors
    labels = [item["labels"] for item in batch]  # list of strings
    return pixel_values, bboxes, labels


# =====================
# 数据加载器
# =====================
dataset = CocoDetectionDataset(DATASET_JSON, IMAGE_DIR, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# =====================
# 优化器
# =====================
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =====================
# 训练循环
# =====================
for epoch in range(EPOCHS):
    epoch_loss = 0
    for pixel_values, bboxes, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        pixel_values = pixel_values.to(DEVICE)

        # 将标签文本拼接成一句话作为 prompt（Grounding DINO 的语言输入）
        # 例如 "vehicle. aircraft. weapon."
        text_prompts = [". ".join(lbls) + "." for lbls in labels]

        # 前向传播
        encoding = processor(images=None, text=text_prompts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model(pixel_values=pixel_values, input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"])

        # 计算 DETR 风格损失
        # Grounding DINO 的输出包含 logits 和 pred_boxes
        # 需要自己计算 loss（如 L1 loss + GIoU loss）
        pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]
        pred_logits = outputs.logits  # [B, num_queries, vocab_size]

        # 简单示例: 仅使用 L1 loss 监督 (实际应结合匹配与分类 loss)
        loss = 0
        for i in range(len(bboxes)):
            gt = bboxes[i].to(DEVICE)
            pred = pred_boxes[i][:len(gt)]  # 匹配前 len(gt) 个预测
            loss += nn.L1Loss()(pred, gt)

        loss /= len(bboxes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}")

    # 每个 epoch 保存模型
    model.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}"))

print("训练完成，模型已保存到:", OUTPUT_DIR)
