import torch
cache = torch.load("datasets_yolo/images/train/train.cache")
print("图片数:", len(cache["images"]))
print("总标签数:", sum(len(x) for x in cache["labels"]))