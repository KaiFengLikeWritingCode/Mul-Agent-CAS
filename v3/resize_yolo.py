from PIL import Image
import json, os

json_path = "./datasets_yolo/instances_val.json"
img_dir = "./datasets_yolo/images/val"

with open(json_path, "r") as f:
    data = json.load(f)

for img in data["images"]:
    img_path = os.path.join(img_dir, img["file_name"])
    if os.path.exists(img_path):
        with Image.open(img_path) as im:
            img["width"], img["height"] = im.size

with open("instances_val_fixed.json", "w") as f:
    json.dump(data, f, indent=2)
