import tensorflow as tf
import json
import re
import numpy as np

MODEL_PATH = "outputs/sentiment/sentiment_model.keras"
WORD_INDEX_PATH = "outputs/sentiment/saved_model/word_index.json"
MAX_LEN = 64

# --- Load Model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load Word Index directly ---
with open(WORD_INDEX_PATH, "r") as f:
    word_index = json.load(f)
assert isinstance(word_index, dict), f"word_index is type {type(word_index)}"

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def encode(text, word_index, max_len=MAX_LEN):
    words = clean_text(text).split()
    seq = [word_index.get(w, 1) for w in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    while len(seq) < max_len:
        seq.append(0)
    return [seq]

for s in ["i love you", "i hate you", "this is awesome", "this is terrible"]:
    encoded = np.array(encode(s, word_index))
    pred = model.predict(encoded)
    print(f"{s!r} -> {pred[0][0]:.4f} ({'Positive' if pred[0][0] > 0.5 else 'Negative'})")
