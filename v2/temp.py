# import requests
#
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
#
# model_id = "IDEA-Research/grounding-dino-base"
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
#
#
#
# # vehicle, aircraft, vessel, weapon, location, other
# #  车辆、飞机、船只、武器、地点、其他


import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())  # 返回True/False

# 进一步信息
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA不可用")