# RNN-SENTIMENT-ANALYSIS

# 🎬 RNN Sentiment Analysis — IMDB Movie Reviews


> A complete NLP pipeline — from raw text to trained RNN — for binary sentiment classification on the IMDB movie review dataset. Built entirely from scratch: text preprocessing, TF-IDF vectorization, a custom PyTorch RNN, and a full training/evaluation loop.



---

##  Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Text Preprocessing](#-text-preprocessing-6-stages)
- [Model Architecture](#-model-architecture)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Design Decisions](#-design-decisions)
- [Further Reading](#-further-reading)

---

## 🧠 Overview

This project builds a binary sentiment classifier (positive / negative) on the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) using a hand-coded Recurrent Neural Network in PyTorch.

The full pipeline covers every stage of a production NLP workflow:

```
Raw Text → Clean → Tokenize → Stem → Vectorize → RNN → Sentiment
```

**No Hugging Face. No pre-trained embeddings. Just NLP fundamentals, built from the ground up.**

This is a strong reference for:
- Understanding classical NLP preprocessing and why each step matters
- Building sequence models with PyTorch's `nn.RNN` from scratch
- Contrasting TF-IDF + RNN with modern embedding-based approaches

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Full NLP Pipeline                         │
│                                                              │
│  IMDB CSV (50K reviews)                                      │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────┐                 │
│  │           Text Preprocessing            │                 │
│  │  Lowercase → URLs → Punctuation →       │                 │
│  │  HTML → Stopwords → Stemming            │                 │
│  └─────────────────────────────────────────┘                 │
│       │                                                      │
│       ▼                                                      │
│  TF-IDF Vectorization (top 5,000 features)                   │
│       │                                                      │
│       ▼                                                      │
│  Train/Test Split (80/20)                                    │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────┐                 │
│  │              RNN Model                  │                 │
│  │  Input(5000) → RNN(128) → FC(1)        │                 │
│  │  + Sigmoid → Binary prediction         │                 │
│  └─────────────────────────────────────────┘                 │
│       │                                                      │
│       ▼                                                      │
│  BCE Loss + Adam → 10 Epochs → Accuracy                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧹 Text Preprocessing — 6 Stages

Each stage removes noise that would otherwise corrupt the model's input signal.

### Stage 1 — Lowercase
```python
df["review"] = df["review"].str.lower()
```
Ensures `"Great"` and `"great"` are treated as the same token.

### Stage 2 — Remove URLs
```python
text = re.sub(r"http\S+", "", text)
```
URLs carry no sentiment signal and inflate vocabulary size.

### Stage 3 — Remove Punctuation
```python
text = re.sub(r"[^A-Za-z0-9\s]", "", text)
```
Strips everything except alphanumeric characters and spaces.

### Stage 4 — Remove HTML Tags
```python
text = re.sub(r"<.*?>", "", text)
```
IMDB reviews often contain raw HTML markup (`<br>`, `<b>`, etc.) from web scraping.

### Stage 5 — Remove Stopwords (NLTK)
```python
stop_words = stopwords.words("english")
# removes: "the", "is", "at", "which", "on" ...
```
Stopwords appear in every document regardless of sentiment — removing them reduces noise and dimensionality.

### Stage 6 — Stemming (Porter Stemmer)
```python
ps = PorterStemmer()
# "running" → "run", "played" → "play", "movies" → "movi"
```
Collapses inflected forms to their root, reducing vocabulary size and helping the model generalize across word variants.

---

## 🧠 Model Architecture

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0  = torch.zeros(num_layers, batch_size, hidden_size)  # initial hidden state
        out, _ = self.rnn(x, h0)      # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])  # take last timestep only
        return out
```

| Layer | Input | Output | Role |
|-------|-------|--------|------|
| `nn.RNN` | (batch, 1, 5000) | (batch, 1, 128) | Sequence processing |
| `nn.Linear` | 128 | 1 | Binary classification head |
| `Sigmoid` | 1 | [0, 1] | Probability output |

**Why `out[:, -1, :]`?**
The RNN processes the sequence left-to-right. The hidden state at the final timestep (`-1`) summarizes the entire input — this is what we pass to the classifier.

---

## 📐 How It Works

### Forward Pass

$$h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$$
$$\hat{y} = \sigma(W_{fc} \cdot h_T + b_{fc})$$

### Loss Function — Binary Cross-Entropy

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

| Component | Choice | Why |
|-----------|--------|-----|
| Loss | `BCELoss` | Binary classification (positive / negative) |
| Optimizer | `Adam` | Adaptive learning rates; robust default |
| Activation | `Sigmoid` | Squashes output to [0,1] probability |
| Threshold | `> 0.5` | Standard binary decision boundary |

---

## 📁 Project Structure

```
📦 rnn-sentiment-analysis/
├── 📓 RNN_for_sentimentanalysis.ipynb   # Full pipeline notebook
├── 📄 IMDB Dataset.csv                  # 50K labeled movie reviews
└── 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis

# Install dependencies
pip install torch pandas scikit-learn nltk
```

### Download NLTK data (first run only)

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### Dataset

Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB Dataset.csv` in the project root.

### Run

```bash
jupyter notebook RNN_for_sentimentanalysis.ipynb
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | IMDB (50K reviews) |
| Train/Test split | 80% / 20% |
| Vocabulary | Top 5,000 TF-IDF features |
| Epochs | 10 |
| Batch size | 64 |
| Hidden size | 128 |
| **Test Accuracy** | **~85%+** *(update with your actual result)* |

> Training completes in under a few minutes on CPU. GPU is not required for this scale.

---

## 🧩 Design Decisions

**Why TF-IDF over word embeddings?**
TF-IDF is interpretable, fast, and surprisingly strong for short-document classification. Word2Vec or GloVe embeddings would capture semantic relationships better, but TF-IDF establishes a clean, understandable baseline — and understanding why embeddings are an improvement requires understanding what TF-IDF is doing first.

**Why Stemming over Lemmatization?**
Porter Stemming is faster and sufficient for this task. Lemmatization (e.g. spaCy) produces linguistically correct root forms but requires POS tagging and adds latency. For a 50K-document corpus, the accuracy difference is marginal.

**Why `nn.RNN` over LSTM or GRU?**
Vanilla RNNs are intentionally used here to expose the fundamental recurrence mechanism without gating complexity. LSTMs solve the vanishing gradient problem for longer sequences — a natural next experiment once the baseline is established.

**Why `unsqueeze(1)` before the RNN?**
`nn.RNN` expects input of shape `(batch, seq_len, input_size)`. Since each review is represented as a single TF-IDF vector (not a true sequence of tokens), we treat it as a sequence of length 1 — hence adding a singleton time dimension with `unsqueeze(1)`.

---

## 🔬 Possible Extensions

| Extension | What it teaches |
|-----------|----------------|
| Replace TF-IDF with GloVe/Word2Vec embeddings | Semantic representation vs. frequency-based |
| Swap `nn.RNN` → `nn.LSTM` or `nn.GRU` | Gated memory and vanishing gradient mitigation |
| Add dropout for regularization | Overfitting control in sequence models |
| Plot loss curves per epoch | Training stability visualization |
| Confusion matrix + precision/recall | Beyond accuracy — class imbalance awareness |

---

## 📚 Further Reading

- [Sutton & Barto — Sequence Modeling with RNNs](https://www.deeplearningbook.org/contents/rnn.html) — Goodfellow et al., Deep Learning Ch. 10
- [NLTK Documentation](https://www.nltk.org/) — tokenization, stopwords, stemming reference
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — vectorizer API
- [PyTorch nn.RNN docs](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — input/output shape reference
- [Original IMDB Dataset Paper](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) — Maas et al., 2011

---

<div align="center">

**Built with PyTorch, NLTK, and 50,000 strongly-opinionated movie fans.**

*Found this useful? Drop a ⭐ to help others discover it.*

</div>

