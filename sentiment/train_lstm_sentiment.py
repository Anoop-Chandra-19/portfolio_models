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
VOCAB_SIZE = 5000  # you can tune this
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

# ---- TOKENIZE ----
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

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
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)  # type: ignore
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
model.export(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# ---- Save tokenizer ----
with open(os.path.join(MODEL_OUT, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer.to_json(), f)
print("Tokenizer config saved.")
