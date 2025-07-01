import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# --------- CONFIG ---------
CLASSES = [
    "apple", "airplane", "book", "bicycle", "banana",
    "aircraft carrier", "alarm clock", "ambulance", "angel", "animal migration",
    "ant", "anvil", "arm", "asparagus", "axe", "backpack", "bandage", "barn",
    "baseball", "baseball bat", "basket", "basketball", "bat", "bathtub",
    "beach", "bear", "bed", "bee", "belt", "bench", "binoculars",
    "bird", "birthday cake", "blackberry", "blueberry", "boomerang", "bottlecap",
    "bowtie", "bracelet", "brain", "bread", "bridge", "broccoli", "broom",
    "bucket", "bulldozer", "The Eiffel Tower", "The Great Wall of China", "The Mona Lisa"
]
SRC_DIR = "doodle/data/raw"
OUT_DIR = "doodle/data/png"
N_IMAGES_PER_CLASS = 2000
IMG_SIZE = 28

def draw_strokes_cv(strokes, size=28, lw=3):
    img = np.ones((size, size), dtype=np.uint8) * 255  # white background
    for stroke in strokes:
        points = list(zip(stroke[0], stroke[1]))
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0] * size / 256), int(points[i][1] * size / 256))
            pt2 = (int(points[i+1][0] * size / 256), int(points[i+1][1] * size / 256))
            cv2.line(img, pt1, pt2, color=(0,), thickness=lw)
    return img

def process_ndjson(ndjson_path, out_dir, n=2000, img_size=28):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(ndjson_path, "r") as f:
        for i, line in enumerate(tqdm(f, total=n, desc=f"{os.path.basename(ndjson_path)}")):
            if i >= n:
                break
            try:
                sample = json.loads(line)
                img = draw_strokes_cv(sample["drawing"], size=img_size)
                cv2.imwrite(os.path.join(out_dir, f"{i}.png"), img)
                count += 1
            except Exception as e:
                print(f"  [WARN] Could not process line {i} in {ndjson_path}: {e}")
    return count

if __name__ == "__main__":
    seen = set()
    summary = []
    for cls in CLASSES:
        if cls in seen:
            print(f"⚠️  Duplicate class detected in CLASSES: {cls} (skipping duplicate)")
            continue
        seen.add(cls)
        ndjson_path = os.path.join(SRC_DIR, f"{cls}.ndjson")
        out_dir = os.path.join(OUT_DIR, cls)
        print(f"\nProcessing {cls}...")
        if not os.path.exists(ndjson_path):
            print(f"  [ERROR] File not found: {ndjson_path} (skipping this class)")
            summary.append((cls, "MISSING"))
            continue
        try:
            num_done = process_ndjson(ndjson_path, out_dir, n=N_IMAGES_PER_CLASS, img_size=IMG_SIZE)
            print(f"  [OK] Saved {num_done} images for class '{cls}'")
            summary.append((cls, f"{num_done} images"))
        except Exception as e:
            print(f"  [ERROR] Failed for {cls}: {e}")
            summary.append((cls, "ERROR"))
    print("\n==== Summary ====")
    for cls, stat in summary:
        print(f"{cls.ljust(24)}: {stat}")
    print("=================")
    print("✅ All classes processed (see warnings/errors above if any).")
