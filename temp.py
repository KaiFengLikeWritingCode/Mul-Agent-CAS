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
python main.py --image datasets/sample_image/12.jpg --text "Tornado of Tactical Air Force Squadron 51 takes off from airbase for mission in Syria. " --out outputs/12_annotated.jpg
python main.py --image datasets/sample_image/16.jpg --text "The Airbus 10 performs a multi-role transport task, carrying soldiers. " --out outputs/16_annotated.jpg
python main.py --image datasets/sample_image/16.jpg --text "The Airbus 10 performs a multi-role transport task, carrying soldiers. " --out outputs/16_annotated.jpg
python main.py --image datasets/sample_image/20.jpg --text "A Luftwaffe Eurofighter flies as Quick Reaction Alert over southern Germany. " --out outputs/20_annotated.jpg
python main.py --image datasets/sample_image/281.jpg --text "A frigate fires a missile of type SM 2. " --out outputs/281_annotated.jpg
python main.py --image datasets/sample_image/15.jpg --text "A Eurofighter in a special livery and another with a different marking fly together with an A400M transport plane and a Transall transport plane in a special retro livery. " --out outputs/15_annotated.jpg
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