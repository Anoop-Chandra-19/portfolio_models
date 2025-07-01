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

### 3. K-Means Clustering Playground
- **Description:** Interactive clustering demo where users add points to a canvas and see live k-means clustering in action.
- **Tech:** K-Means clustering algorithm.
- **Demo:** Users can experiment with clustering by adding/removing points and changing the number of clusters.

---

## 📁 Project Structure

```
sentiment/         # Sentiment analysis model (TinyBERT)
doodle/            # Doodle classification model (QuickDraw CNN)
playground/        # K-means clustering demo
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
   pip install -r requirements.txt
   ```

3. **Train models:**
   - Sentiment: See [`sentiment/train_tinybert.py`](sentiment/train_tinybert.py)
   - Doodle: See [`doodle/train_cnn.py`](doodle/train_cnn.py)
   - K-Means: See [`playground/kmeans_prototype.py`](playground/kmeans_prototype.py)

4. **Export models for web:**
   - Use scripts in `export_tf.py` or `utils/convert_to_tfjs.py` as needed.

---

## 🌐 Web Integration

- Models are exported in web-friendly formats (e.g., TensorFlow.js, ONNX, or custom JS).
- Each model is integrated into a card on the portfolio website for live demo and interaction.

---

## 📊 Performance

- **Sentiment Analysis:** [87.4]
- **Doodle Classification:** [Add accuracy here]
- **K-Means Playground:** Real-time clustering in browser

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

Anoop  