import os
import shutil
import random
from glob import glob

# --------- CONFIG ---------
SRC_DIR = "doodle/data/png"
DST_DIR = "doodle/data/split"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

random.seed(SEED)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_copy(class_dir, class_name, dst_dir, ratios):
    images = glob(os.path.join(class_dir, "*.png"))
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * ratios["train"])
    n_val = int(n_total * ratios["val"])
    # remainder is assigned to test set

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(dst_dir, split, class_name)
        ensure_dir(split_dir)
        for img in files:
            shutil.copy(img, os.path.join(split_dir, os.path.basename(img)))
    return {split: len(files) for split, files in splits.items()}

if __name__ == "__main__":
    classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
    print(f"Found {len(classes)} classes.")

    summary = {}
    for cls in classes:
        class_dir = os.path.join(SRC_DIR, cls)
        stats = split_and_copy(class_dir, cls, DST_DIR, SPLIT_RATIOS)
        summary[cls] = stats
        print(f"{cls:24}: train={stats['train']}, val={stats['val']}, test={stats['test']}")

    print("\n==== Dataset Split Summary ====")
    for cls, stats in summary.items():
        print(f"{cls:24}: train={stats['train']}, val={stats['val']}, test={stats['test']}")
    print("==============================")
    print("✅ All classes split and copied.")
