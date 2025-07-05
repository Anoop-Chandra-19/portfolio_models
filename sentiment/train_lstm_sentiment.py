import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional 
from tensorflow.keras.models import Sequential                                    
from tensorflow.keras.preprocessing.text import Tokenizer                          
from tensorflow.keras.preprocessing.sequence import pad_sequences                  

# ---- CONFIG ----
DATA_DIR = "sentiment/data"
MODEL_OUT = "outputs/sentiment/saved_model"
VOCAB_SIZE = 10000  # you can tune this
MAX_LEN = 64       # you can tune this
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 6

# ---- DATA LOAD ----
def load_data(filename):
    df = pd.read_csv(filename)
    texts = df['review'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train.csv"))
val_texts, val_labels     = load_data(os.path.join(DATA_DIR, "val.csv"))
test_texts, test_labels   = load_data(os.path.join(DATA_DIR, "test.csv"))

# --- DATA AUGMENTATION: Add clear positive/negative sentences ---
custom_sentences = [
    # NEGATIVE
    ("i hate you", 0),
    ("i hate this", 0),
    ("i hate it", 0),
    ("hate this", 0),
    ("hate it", 0),
    ("i dislike this", 0),
    ("this is terrible", 0),
    ("worst movie ever", 0),
    ("this is bad", 0),
    ("so bad", 0),
    ("awful experience", 0),
    ("this made me angry", 0),
    ("absolutely horrible", 0),
    ("i'm very disappointed", 0),
    ("i can't stand this", 0),
    ("i am sad", 0),
    ("i feel sad", 0),
    ("this sucks", 0),
    ("not good", 0),
    ("do not recommend", 0),
    ("waste of time", 0),
    ("what a letdown", 0),
    ("very boring", 0),
    ("i regret watching", 0),
    ("such a bad movie", 0),
    ("it was painful to watch", 0),
    ("never again", 0),
    ("so boring", 0),
    ("not worth it", 0),
    ("big disappointment", 0),

    # POSITIVE
    ("i love you", 1),
    ("i love this", 1),
    ("i love it", 1),
    ("love this", 1),
    ("love it", 1),
    ("i like this", 1),
    ("this is awesome", 1),
    ("best movie ever", 1),
    ("this is good", 1),
    ("so good", 1),
    ("i am happy", 1),
    ("wonderful experience", 1),
    ("this made me smile", 1),
    ("absolutely fantastic", 1),
    ("i'm very impressed", 1),
    ("i enjoyed this", 1),
    ("it was great", 1),
    ("this is amazing", 1),
    ("superb", 1),
    ("highly recommend", 1),
    ("worth every minute", 1),
    ("what a treat", 1),
    ("so entertaining", 1),
    ("i'd watch it again", 1),
    ("such a good movie", 1),
    ("i laughed so much", 1),
    ("pure joy", 1),
    ("it made my day", 1),
    ("very fun", 1),
]

# Convert to DataFrame and add to train set
custom_df = pd.DataFrame(custom_sentences, columns=["review", "label"])
aug_train_texts = train_texts + custom_df["review"].tolist()
aug_train_labels = train_labels + custom_df["label"].tolist()
print(f"Augmented training set size: {len(aug_train_texts)}")

# ---- TOKENIZE ----
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
tokenizer.fit_on_texts(aug_train_texts)

print("Top 20 words in tokenizer:", list(tokenizer.word_index.items())[:20])
important_words = ["i", "love", "hate", "you", "sad", "movie", "very", "this"]
for word in important_words:
    print(f"Index for '{word}':", tokenizer.word_index.get(word))
print("OOV token:", tokenizer.oov_token)
print("OOV index:", tokenizer.word_index.get(tokenizer.oov_token))
print("Tokenizer vocab size (should be <= VOCAB_SIZE):", len(tokenizer.word_index))

def encode(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

X_train = encode(train_texts)
X_val   = encode(val_texts)
X_test  = encode(test_texts)
y_train = np.array(train_labels)
y_val   = np.array(val_labels)
y_test  = np.array(test_labels)

# ---- MODEL ----
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', unroll=True)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---- EVALUATE ----
loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"Test accuracy: {acc:.4f}")

# ---- EXPORT ----
os.makedirs(MODEL_OUT, exist_ok=True)

# Save Keras model (.keras) for Python
keras_path = os.path.join("outputs/sentiment", "sentiment_model.keras")
model.save(keras_path)
print(f"Keras model saved at {keras_path}")

# Export for TensorFlow.js (SavedModel format)
model.export(MODEL_OUT)
print(f"Model exported to {MODEL_OUT}")

# Save tokenizer config (for reference/legacy)
tokenizer_config_path = os.path.join(MODEL_OUT, "tokenizer_config.json")
with open(tokenizer_config_path, "w") as f:
    f.write(tokenizer.to_json())
print(f"Tokenizer config saved at {tokenizer_config_path}")

# Save word_index as clean JSON for JS/Python
word_index_path = os.path.join(MODEL_OUT, "word_index.json")
with open(word_index_path, "w") as f:
    json.dump(tokenizer.word_index, f)
print(f"Word index saved at {word_index_path}")
