import json
import os
from PIL import Image

input_json = "./datasets/sample_entity.json"
dataset_dir = "./datasets/sample_image"
output_json = "coco_annotations.json"

# 定义类别
category_list = ["aircraft", "vehicle", "weapon", "vessel", "location", "other"]

with open(input_json, "r", encoding="utf-8") as f:
    annotations = json.load(f)

coco = {"images": [], "annotations": [], "categories": []}
categories = {cat: i+1 for i, cat in enumerate(category_list)}

for cat, cid in categories.items():
    coco["categories"].append({"id": cid, "name": cat})

ann_id = 1

for img_id, entities in annotations.items():
    img_path = os.path.join(dataset_dir, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_path}")
        continue

    img = Image.open(img_path)
    width, height = img.size

    coco["images"].append({
        "id": int(img_id),
        "file_name": f"{img_id}.jpg",
        "width": width,
        "height": height
    })

    for ent in entities:
        bnd = ent["bnd"]
        if bnd:  # 仅对有bbox的实体生成标注
            xmin, ymin, xmax, ymax = bnd["xmin"], bnd["ymin"], bnd["xmax"], bnd["ymax"]
            w, h = xmax - xmin, ymax - ymin
            cat_id = categories.get(ent["label"], None)
            if cat_id:
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

# 保存 COCO JSON
os.makedirs("annotations", exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)
print(f"✅ COCO annotations saved to: {output_json}")