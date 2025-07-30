import json
import os
from tqdm import tqdm

def coco_to_yolo(json_path, img_dir, out_dir):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    os.makedirs(out_dir, exist_ok=True)

    img_id_map = {img["id"]: img for img in coco["images"]}

    for ann in tqdm(coco["annotations"], desc=f"Converting {json_path}"):
        img_info = img_id_map[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = ann["bbox"]

        # 转换为YOLO格式 (cx, cy, w, h) 归一化
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        label_path = os.path.join(out_dir, img_info["file_name"].replace(".jpg", ".txt"))
        with open(label_path, 'a') as f_out:
            f_out.write(f"{ann['category_id']-1} {cx} {cy} {nw} {nh}\n")  # -1转为0索引

coco_to_yolo("datasets_yolo/annotations/instances_train.json",
             "datasets_yolo/images/train",
             "datasets_yolo/labels/train")

coco_to_yolo("datasets_yolo/annotations/instances_val.json",
             "datasets_yolo/images/val",
             "datasets_yolo/labels/val")
