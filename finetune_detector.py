import os

import torch
import argparse
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.misc import nested_tensor_from_tensor_list

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from GINO_dataset import COCODetectionDataset

# img_folder = "./datasets/images"
# ann_file = "./datasets/annotations/coco_annotations.json"
# train_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.Resize((800, 800)),
#     transforms.ToTensor(),
# ])
# # 创建数据集
# dataset_train = COCODetectionDataset(img_folder=img_folder, ann_file=ann_file, transforms=train_transforms)
# dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))

# === 训练函数 ===
def train(args):
    import pkg_resources

    # 自动定位 config 文件
    cfg_path = pkg_resources.resource_filename("groundingdino", "config/GroundingDINO_SwinT_OGC.py")
    config = SLConfig.fromfile(cfg_path)

    model = build_model(config).cuda()
    model.train()

    # 冻结 backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # 数据集
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.Resize((800, 800)),
        transforms.ToTensor()
    ])

    dataset = COCODetectionDataset(
        img_folder=os.path.join(args.dataset, "images"),
        ann_file=os.path.join(args.dataset, "annotations/coco_annotations.json"),
        transforms=transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for imgs, targets in dataloader:
            imgs = nested_tensor_from_tensor_list([i.cuda() for i in imgs])
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs, targets)  # 新版模型直接返回损失字典
                loss = outputs["loss_ce"] + outputs["loss_bbox"] + outputs["loss_giou"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "groundingdino_finetuned.pth")
    print("✅ 微调完成并保存模型：groundingdino_finetuned.pth")

# === 主程序入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--dataset", default="datasets")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args)