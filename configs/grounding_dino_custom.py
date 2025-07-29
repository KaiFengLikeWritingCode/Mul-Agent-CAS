# configs/grounding_dino_custom.py
_base_ = "grounding_dino_R50_OGC.py"

# 1) 数据集注册
DATASETS = dict(
    TRAIN=("coco_custom_train",),   # 自定义训练集
    TEST=(),                        # 暂不做验证
)
DATASETS.ANNOTATION_FILES = dict(
    coco_custom_train="coco_annotations2.json",
)

# 2) 图片目录
#    请确保这里指向你的原始图片文件夹
DATASETS.PATHS = dict(
    COCO_IMAGES_DIR="./datasets/sample_image",
)

# 3) 模型权重初始化
MODEL = dict(
    WEIGHTS="https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/pytorch_model.bin"
)

# 4) 训练超参（根据显存自行调节）
SOLVER = dict(
    IMS_PER_BATCH=8,    # per-GPU batch size
    BASE_LR=2e-5,
    MAX_ITER=30000,     # 总 iteration 数
    STEPS=(20000, 28000),
)

# 5) 日志 & 输出
OUTPUT_DIR = "./output_custom"
