# batch_process.py

import json
import os
import argparse
from pathlib import Path

from main import process, draw_and_save  # draw_and_save 绘制并保存带框图像

def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}

def save_json_atomic(obj: dict, path: Path):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def bbox_to_dict(bbox):
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)}

def main(text_json, image_dir, out_json, annotated_dir):
    text_json    = Path(text_json)
    image_dir    = Path(image_dir)
    out_json     = Path(out_json)
    annotated_dir = Path(annotated_dir)

    samples   = load_json(text_json)
    processed = load_json(out_json)
    total     = len(samples)

    for idx, (img_id, info) in enumerate(sorted(samples.items(), key=lambda x: int(x[0])), 1):
        if img_id in processed:
            print(f"[{idx}/{total}] {img_id} already processed, skip.")
            continue

        text     = info.get("text", "").strip()
        img_path = image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"[{idx}/{total}] {img_id}: image not found, skipping.")
            processed[img_id] = []
            save_json_atomic(processed, out_json)
            continue

        try:
            # 调用主流程
            img, results = process(str(img_path), text)

            # 转换实体结果
            entries = []
            for r in results:
                entries.append({
                    "name":  r["name"],
                    "label": r["label"],
                    "bnd":   bbox_to_dict(r.get("bbox")),
                })

            # 可视化保存带框图像
            annotated_path = annotated_dir / f"{img_id}.jpg"
            draw_and_save(img.copy(), results, str(annotated_path))

        except Exception as e:
            print(f"[{idx}/{total}] {img_id} error: {e!r}")
            entries = []

        # 写入 JSON 并持久化
        processed[img_id] = entries
        save_json_atomic(processed, out_json)
        print(f"[{idx}/{total}] {img_id} done, {len(entries)} entities, saved JSON and image.")

    print("Batch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_json",    type=str, default="./datasets/sample_text.json")
    parser.add_argument("--image_dir",    type=str, default="./datasets/sample_image")
    parser.add_argument("--out_json",     type=str, default="./outputs/sample_entity.json")
    parser.add_argument("--annotated_dir",type=str, default="./outputs/annotated")
    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.annotated_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    main(
        text_json     = args.text_json,
        image_dir     = args.image_dir,
        out_json      = args.out_json,
        annotated_dir = args.annotated_dir,
    )
