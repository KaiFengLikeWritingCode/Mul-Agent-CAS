import json
from pathlib import Path
from collections import defaultdict

JSON_IN   = Path("./datasets_0/sample_entity.json")
IMG_DIR   = Path("./datasets_0/sample_image")
JSON_OUT  = Path("coco_annotations2.json")

# 读取原始标注
with JSON_IN.open("r", encoding="utf-8") as f:
    raw = json.load(f)

# 用来收集不为空的图片 id 及其 annotations
images = []
annotations = []
label2id = {}
next_cat_id = 1
ann_id = 1

for idx, ents in raw.items():
    # 过滤：如果 ents 为空列表，或者所有 bnd 都是 None，跳过
    valid = [e for e in ents
             if e.get("bnd")
             and all(k in e["bnd"] for k in ("xmin","ymin","xmax","ymax"))]
    if not valid:
        continue

    # 确认图片存在
    img_path = IMG_DIR / f"{idx}.jpg"
    if not img_path.exists():
        print(f"跳过: 找不到图片 {img_path}")
        continue

    # 添加 image 信息
    images.append({
        "id": int(idx),
        "file_name": f"{idx}.jpg",
        "height": None,  # 可选：运行时填充 or 留空
        "width":  None,
    })

    # 遍历有效实体
    for e in valid:
        label = e["label"]
        # 给每个 label 分配一个 category_id
        if label not in label2id:
            label2id[label] = next_cat_id
            next_cat_id += 1

        b = e["bnd"]
        # COCO bbox 格式： [x, y, width, height]
        x, y = b["xmin"], b["ymin"]
        w = b["xmax"] - b["xmin"]
        h = b["ymax"] - b["ymin"]

        annotations.append({
            "id": ann_id,
            "image_id": int(idx),
            "category_id": label2id[label],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

# 构建 categories 列表
categories = [
    {"id": cid, "name": lbl}
    for lbl, cid in label2id.items()
]

# 写出 COCO 格式
coco_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with JSON_OUT.open("w", encoding="utf-8") as f:
    json.dump(coco_dict, f, ensure_ascii=False, indent=2)

print(f"已生成 COCO 格式标注：{JSON_OUT}")
print(f"  images:      {len(images)} 张")
print(f"  annotations: {len(annotations)} 条")
print(f"  categories:  {len(categories)} 种")