# GINO_dataset.py

from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class COCODetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.transforms = transforms
        # 类别 ID 到名称的映射
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}

    def __getitem__(self, index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_folder, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        # 跳过无标注样本
        if len(labels) == 0:
            return self.__getitem__((index + 1) % len(self))



        # 转 tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 生成 caption（确保非空）
        caption_tokens = [self.cat_id_to_name[l.item()] for l in labels if l.item() in self.cat_id_to_name]
        if len(caption_tokens) == 0:
            return self.__getitem__((index + 1) % len(self))

        caption = ", ".join(caption_tokens) if caption_tokens else "object"

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "caption": caption
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)
