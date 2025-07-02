# Portfolio Machine Learning Models

This repository contains machine learning models trained and exported for live interactive demos on my portfolio website. Each model is showcased with a web-based demo card, allowing users to try them out in real time.

---

## 🚀 Models Overview

### 1. Sentiment Analysis (TinyBERT)
- **Description:** Classifies user-input text as positive or negative sentiment.
- **Tech:** Fine-tuned TinyBERT model.
- **Demo:** Users can type any sentence and instantly see the sentiment prediction.

### 2. Doodle Classification
- **Description:** Recognizes hand-drawn doodles using a model trained on the Google QuickDraw dataset.
- **Tech:** Convolutional Neural Network (CNN).
- **Demo:** Users can draw on a canvas, and the model predicts the doodle category.

---

## 📁 Project Structure

```
sentiment/         # Sentiment analysis model (TinyBERT)
doodle/            # Doodle classification model (QuickDraw CNN)
utils/             # Utility scripts (preprocessing, conversion, etc.)
outputs/           # Exported models and checkpoints
```

---

## 🛠️ Setup & Training

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd portfolio_models
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Train models:**
   - Sentiment: See [`sentiment/train_lstm_sentiment.py`](sentiment/train_lstm_sentiment.py)
   - Doodle: See [`doodle/train_cnn.py`](doodle/train_cnn.py)

4. **Export models for web:**
   - Use script in [`utils/convert_to_tfjs.py`](utils/convert_to_tfjs.py) as needed.

---

## 🧩 Scripts Overview

### Doodle Scripts (`doodle/`)
- [`quickdraw_to_png.py`](doodle/quickdraw_to_png.py): Converts QuickDraw `.ndjson` files to PNG images for training.
- [`split_png_dataset.py`](doodle/split_png_dataset.py): Splits the PNG dataset into training, validation, and test sets.
- [`train_cnn.py`](doodle/train_cnn.py): Trains the CNN model for doodle classification.

### Sentiment Scripts (`sentiment/`)
- [`prep_imdb_data.py`](sentiment/prep_imdb_data.py): Prepares and splits the IMDB dataset for sentiment analysis.
- [`train_lstm_sentiment.py`](sentiment/train_lstm_sentiment.py): Trains an LSTM-based sentiment analysis model.

### Utility Scripts (`utils/`)
- [`convert_to_tfjs.py`](utils/convert_to_tfjs.py): Converts trained models to TensorFlow.js format for web deployment.
- [`preprocess_doodle.py`](utils/preprocess_doodle.py): Preprocessing utilities for doodle data.

---

## 🌐 Web Integration

- Models are exported in web-friendly formats (TensorFlow.js).
- Each model is integrated into a card on the portfolio website for live demo and interaction.

---

## 📊 Performance

- **Sentiment Analysis:** [80%] for 2 divisional classes
- **Doodle Classification:** [63%] for 50 divisional classes

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

Anoop