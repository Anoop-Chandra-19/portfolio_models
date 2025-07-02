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
   - Sentiment: See [`sentiment/train_tinybert.py`](sentiment/train_tinybert.py)
   - Doodle: See [`doodle/train_cnn.py`](doodle/train_cnn.py)

4. **Export models for web:**
   - Use script in `utils/convert_to_tfjs.py` as needed.

---

## 🌐 Web Integration

- Models are exported in web-friendly formats (e.g., TensorFlow.js, ONNX, or custom JS).
- Each model is integrated into a card on the portfolio website for live demo and interaction.

---

## 📊 Performance

- **Sentiment Analysis:** [87.4]
- **Doodle Classification:** [Add accuracy here]

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

Anoop  