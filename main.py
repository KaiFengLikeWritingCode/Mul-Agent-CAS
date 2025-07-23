import argparse
from PIL import Image
from ner import extract_entities
from detector import detect_boxes
from matcher import match_entities

def crop_regions(image: Image.Image, boxes):
    crops = []
    for xmin, ymin, xmax, ymax in boxes:
        crops.append(image.crop((xmin, ymin, xmax, ymax)).convert("RGB"))
    return crops

def process(image_path, text):
    ents = extract_entities(text)
    all_results = []
    img = Image.open(image_path)
    for ent in ents:
        boxes = detect_boxes(image_path, ent["name"])
        if not boxes:
            continue
        crops = crop_regions(img, boxes)
        matched = match_entities([ent], crops, boxes)
        all_results.extend(matched)
    return all_results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--text",  type=str, required=True)
    args = p.parse_args()

    results = process(args.image, args.text)
    print("识别结果：")
    for r in results:
        print(r)
