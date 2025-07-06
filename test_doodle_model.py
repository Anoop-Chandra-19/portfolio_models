import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# ---- CONFIG ----
KERAS_OUT = "outputs/doodle/doodle_cnn_model.keras"
CLASS_NAMES_PATH = "outputs/doodle/saved_model/class_names.json"
DATA_DIR = "doodle/data/split"
IMG_SIZE = 28
BATCH_SIZE = 32

# ---- LOAD MODEL & CLASS NAMES ----
model = tf.keras.models.load_model(KERAS_OUT)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
print(f"Loaded model from {KERAS_OUT}")
print(f"Classes: {class_names}")

# ---- GET SAMPLE IMAGES FROM TEST SET ----

def get_random_image_from_class(class_name):
    class_dir = os.path.join(DATA_DIR, "test", class_name)
    files = os.listdir(class_dir)
    img_file = random.choice(files)
    img_path = os.path.join(class_dir, img_file)
    return img_path

sample_categories = random.sample(class_names, min(6, len(class_names)))
sample_images = [(get_random_image_from_class(c), c) for c in sample_categories]

print("\nSample images for testing:")
for img_path, category in sample_images:
    print(f"  - {img_path} (label: {category})")

def load_grayscale_image_pil(path, img_size):
    img = load_img(path, color_mode='grayscale', target_size=(img_size, img_size))
    arr = img_to_array(img)             # shape (28, 28, 1), dtype float32, values [0, 255]
    arr = np.expand_dims(arr, axis=0)   # shape (1, 28, 28, 1)
    return arr

for img_path, true_label in sample_images:
    img = load_grayscale_image_pil(img_path, 28)
    print("Loaded image shape for prediction:", img.shape)
    print("Pixel stats: min", img.min(), "max", img.max(), "mean", img.mean())
    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.show()
    preds = model.predict(img)
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3_names = [class_names[i] for i in top3_idx]
    top3_scores = [preds[0][i] for i in top3_idx]
    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"  True label: {true_label}")
    print(f"  Top 3 predicted: {top3_names} (scores: {[f'{s:.2f}' for s in top3_scores]})")
