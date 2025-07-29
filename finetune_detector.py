import torch
import argparse
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from GINO_dataset import COCODetectionDataset

img_folder = "./datasets/images"
ann_file = "./datasets/annotations/coco_annotations.json"
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])
# 创建数据集
dataset_train = COCODetectionDataset(img_folder=img_folder, ann_file=ann_file, transforms=train_transforms)
dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: list(zip(*x)))

def train(args):
    # 1. 加载配置和模型
    config = SLConfig.fromfile("groundingdino/config/GroundingDINO_SwinT_OGC.py")
    model, criterion, _ = build_model(config)
    model = model.cuda()

    # 2. 冻结 backbone（小数据集防过拟合）
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False



    # 4. 优化器 & AMP
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 5. 训练循环
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for images, targets in dataloader_train:
            images = nested_tensor_from_tensor_list([img.cuda() for img in images])
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {epoch_loss/len(dataloader_train):.4f}")

        # 每隔5个epoch解冻高层backbone进一步finetune
        if epoch == 5:
            print("🔓 解冻高层 backbone 进行进一步微调...")
            for name, param in model.named_parameters():
                if "backbone.layers.3" in name:  # 解冻最后一层block
                    param.requires_grad = True

    # 保存模型
    torch.save(model.state_dict(), "groundingdino_finetuned.pth")
    print("✅ 模型已保存：groundingdino_finetuned.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", default="datasets", help="COCO格式数据集路径")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args)
