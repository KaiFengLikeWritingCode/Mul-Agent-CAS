# test_detector_api.py

import argparse
import os
from PIL import Image, ImageDraw
from detector_api import detect_boxes

def draw_boxes_on_image(image_path: str, boxes, output_path: str):
    """
    简单地把 detect_boxes 输出的 rectangles 画到原图上，并保存到 output_path。
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Test detector_api.detect_boxes() with an image and a phrase"
    )
    parser.add_argument(
        "--image", "-i", required=True, default="datasets/sample_image/15.jpg",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--phrase", "-p", required=True,default="Eurofighter",
        help="Text phrase to ground (e.g. 'Eurofighter')"
    )
    parser.add_argument(
        "--out", "-o", default="outputs/test_annotated_15.jpg",
        help="Where to save the annotated image (optional)"
    )
    args = parser.parse_args()

    print(f"Running detect_boxes on image: {args.image!r} with phrase: {args.phrase!r}")
    boxes = detect_boxes(args.image, args.phrase)
    if not boxes:
        print("No boxes detected.")
    else:
        print("Detected boxes:")
        for idx, box in enumerate(boxes):
            print(f"  [{idx}] {box}")

    # 可视化
    draw_boxes_on_image(args.image, boxes, args.out)

if __name__ == "__main__":
    main()
