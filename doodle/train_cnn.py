import os
import json
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
import shutil
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras import layers, Sequential # type: ignore

# ---- CONFIG ----
DATA_DIR = "doodle/data/split"
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 24
KERAS_OUT = "outputs/doodle/doodle_cnn_model.keras"
SAVEDMODEL_DIR = "outputs/doodle/saved_model"

# ---- DATA LOAD ----
def get_datasets():
    train_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )
    val_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names

train_ds, val_ds, test_ds, class_names = get_datasets()
NUM_CLASSES = len(class_names)
print(f"Loaded {NUM_CLASSES} classes: {class_names}")

# ---- MODEL ----
# Data augmentation for training only
data_augmentation = tf.keras.Sequential([   # type: ignore
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08, 0.08),
])

def augment(images, labels):
    return data_augmentation(images, training=True), labels

train_ds_aug = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds_aug = train_ds_aug.prefetch(tf.data.AUTOTUNE)

# Build model WITHOUT augmentation layers
def build_improved_cnn(num_classes, img_size=28):
    model = Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

model = build_improved_cnn(NUM_CLASSES)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ---- TRAIN ----
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True) # type: ignore
]
# Train using AUGMENTED training data
history = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---- EVALUATE ----
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# ---- SAVE MODEL ----
os.makedirs(os.path.dirname(KERAS_OUT), exist_ok=True)

# Save as Keras format (optional, good for restoring weights)
model.save(KERAS_OUT)
print(f"Model saved to {KERAS_OUT}")

# Export as TensorFlow SavedModel (for TF.js)
# Remove previous export if it exists to avoid Keras export error
if os.path.exists(SAVEDMODEL_DIR):
    shutil.rmtree(SAVEDMODEL_DIR)
model.export(SAVEDMODEL_DIR)
print(f"SavedModel exported to {SAVEDMODEL_DIR}")

# ---- Save label mapping ----
with open(os.path.join(SAVEDMODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)
print("Class names saved in SavedModel directory.")

# (Optional) Also save class names in Keras model's folder:
with open(os.path.join(os.path.dirname(KERAS_OUT), "class_names.json"), "w") as f:
    json.dump(class_names, f)
print("Class names saved in Keras directory.")