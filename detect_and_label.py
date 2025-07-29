import json
import os
from PIL import Image, ImageDraw, ImageFont

# 加载标注文件
json_path = "./datasets/sample_entity.json"
with open(json_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# 数据集图片目录（需确认路径是否正确）
dataset_dir = "./datasets/sample_image"  # 存放 idx.jpg 图片的文件夹
output_dir = "outputs/labeled_images"
os.makedirs(output_dir, exist_ok=True)

# 字体设置
try:
    font = ImageFont.truetype("arial.ttf", size=18)
except IOError:
    font = ImageFont.load_default()

# 遍历标注并绘制
for idx, entities in annotations.items():
    image_path = os.path.join(dataset_dir, f"{idx}.jpg")
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        continue

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ent in entities:
        name = ent["name"]
        label = ent["label"]
        bnd = ent["bnd"]

        if bnd:  # 有bbox才绘制
            xmin, ymin, xmax, ymax = bnd["xmin"], bnd["ymin"], bnd["xmax"], bnd["ymax"]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            text = f"{name} ({label})"

            # 绘制文字背景
            tb = draw.textbbox((xmin, ymin), text, font=font)
            draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill="red")
            draw.text((xmin, ymin), text, fill="white", font=font)

    out_path = os.path.join(output_dir, f"{idx}_annotated.jpg")
    img.save(out_path)
    print(f"✅ Annotated image saved: {out_path}")
