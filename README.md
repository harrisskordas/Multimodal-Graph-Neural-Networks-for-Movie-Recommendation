# 🎬 Multimodal Graph Neural Networks for Movie Recommendation

> **Deep Learning Lab Project — Part 2** | Group 33 | DSML, NTUA | 2025–2026

A three-stage multimodal pipeline that combines **Computer Vision**, **Natural Language Processing**, and **Graph Neural Networks** to build an intelligent movie recommendation system using the MovieLens dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Task 1: Computer Vision](#task-1-computer-vision)
- [Task 2: Natural Language Processing](#task-2-natural-language-processing)
- [Task 3: Graph Neural Network](#task-3-graph-neural-network)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Repository Structure](#repository-structure)
- [Team](#team)

---

## Overview

This project builds a **content-aware recommendation system** by extracting multimodal features from movie metadata and fusing them within a graph learning framework.

**Core idea:**
1. **Vision** — Learn what a movie *looks like* from its poster → 512-dim embedding
2. **Language** — Learn what a movie *is about* from its plot summary → 128-dim embedding
3. **Graph** — Learn which movies a *user* will enjoy by propagating these embeddings through a bipartite user–movie interaction graph

```
Poster Image  ──► ResNet-18 ────────────────────────┐
                                                     ├──► Concatenate ──► GNN ──► Rating Prediction
Plot Summary  ──► RNN + Attention + GloVe ──────────┘
```

---

## Architecture

### Data Pipeline

| Source | Description | Size |
|--------|-------------|------|
| `ml-25m` | Training movies (stratified sample) | 15,000 movies |
| `ml-latest-small` | GNN target graph | ~9,000 movies, 600 users, 100K ratings |
| TMDB API | Posters + plot summaries | Async fetch (30 concurrent) |

**Stratified sampling** ensures the training distribution of decade × primary-genre matches the downstream evaluation graph, preventing distributional mismatch.

---

## Task 1: Computer Vision

### Problem
Multi-label genre classification from poster images ($128 \times 128$ px) — predict a binary vector over 20 genres.

### Models

#### Custom CNN (from scratch)

```
Input (3×128×128)
  → Conv2d(3, 32, 3) + ReLU + MaxPool2d(2)
  → Conv2d(32, 64, 3) + ReLU + MaxPool2d(2)
  → Conv2d(64, 128, 3, stride=2) + ReLU + MaxPool2d(2)   ← stride=2 reduces overfitting
  → Flatten → Linear(8192, 512) → Dropout(0.65)
  → Linear(512, 20)   [raw logits]
```

**Key design choice:** stride=2 in the 3rd conv layer was selected after observing that stride=1 caused overfitting within 20 epochs — the larger stride reduces the number of dense layer parameters and acts as implicit regularisation.

#### Pre-trained ResNet-18 (fine-tuned)

- Backbone: ResNet-18 pre-trained on ImageNet
- Modified head: `AdaptiveAvgPool2d → Dropout(0.7) → Linear(512, 20)`
- All layers frozen except final block + classification head

### Training

| Hyperparameter | Custom CNN | ResNet-18 |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning Rate | 1e-5 | 1e-4 |
| Epochs | 25 | 10 |
| Batch Size | 32 | 32 |
| Loss | BCEWithLogitsLoss + pos_weight | BCEWithLogitsLoss + pos_weight |
| Dropout | 0.65 | 0.7 |

**Class imbalance handling:**
$$w_c = \sqrt{\frac{N - N_c^+}{N_c^+}}$$

### Results

| Model | Macro F1 | Micro F1 | ROC-AUC |
|-------|----------|----------|---------|
| Custom CNN | 0.1983 | 0.4343 | 0.6644 |
| **ResNet-18** | **0.2736** | **0.4748** | **0.7124** |

ResNet-18 outperforms across all metrics. Transfer learning from ImageNet provides representations that cannot be matched by training from scratch on 15K posters.

### Embedding Extraction

```python
feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])
# Output: [N, 512] tensor per movie (penultimate layer)
torch.save({'movie_ids': ..., 'embeddings': final_visual_embeddings}, 'cv_embeddings.pt')
```

---

## Task 2: Natural Language Processing

### Problem
Same multi-label classification, but from **plot summaries** (text).

### Preprocessing

```python
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# Vocabulary: top 10,000 tokens (training set only)
# Sequence length: 150 tokens (truncate/pad)
```

### Model Architecture

```
Input: token IDs [Batch, 150]
  → nn.Embedding(10002, 100) ← initialised with GloVe 6B 100d
  → Dropout(0.4)
  → nn.RNN(100, 128, batch_first=True)   # all hidden states H = [h1...h150]
  → Attention:
      e_t = W_a · h_t          # score per token
      α = softmax(e)           # normalised weights
      c = Σ α_t h_t            # context vector [128]
  → Dropout(0.5)
  → Linear(128, 20)            # raw logits
```

**Self-Attention benefits:**
- Focuses on genre-relevant tokens (e.g., "cowboy", "outlaw" → Western)
- Mitigates padding noise in variable-length sequences
- Improves interpretability

### GloVe Integration

```python
# Freeze embeddings for 3 warmup epochs, then fine-tune
nlp_model.embedding.weight.requires_grad = not freeze_embeddings
# Vocabulary coverage: ~87%
```

### Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, lr=1e-3, weight_decay=1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Gradient Clipping | max_norm=3.0 |
| Epochs | 25 (base) + 10 (GloVe) |
| Unfreeze GloVe at | Epoch 3 |

### Results

| Model | Macro F1 | Micro F1 | ROC-AUC |
|-------|----------|----------|---------|
| NLP (RNN + Attention) | 0.2054 | 0.3201 | 0.6054 |
| CV (ResNet-18) | 0.1983 | 0.4343 | **0.6609** |

CV outperforms NLP on Micro F1 and ROC-AUC. Movie posters contain explicit visual genre cues (colour, composition), while plot summaries require deeper semantic understanding.

### Embedding Extraction

```python
context_vector = nlp_model.extract_embeddings(texts)
# Output: [N, 128] tensor per movie
torch.save({'movie_ids': ..., 'embeddings': final_nlp_embeddings}, 'nlp_embeddings.pt')
```

---

## Task 3: Graph Neural Network

### Graph Construction

```python
# Bipartite heterogeneous graph
data['user'].num_nodes = len(user_mapping)      # ~600 users
data['movie'].x = movie_features                # ~9K movies
data['user', 'rates', 'movie'].edge_index = edge_index  # ~100K edges

# Undirected for bidirectional message passing
data = T.ToUndirected()(data)
```

### Model: Heterogeneous GraphSAGE

```
User nodes:  nn.Embedding(num_users, 64)           ← learnable structural IDs
Movie nodes: nn.Linear(feat_dim, 64)               ← project content features

HeteroConv Layer 1: SAGEConv((-1,-1), 64) × 2 edge types + ReLU
HeteroConv Layer 2: SAGEConv((-1,-1), 64) × 2 edge types

Decoder:
  z_u || z_m  →  Linear(128, 64)  →  ReLU  →  Linear(64, 1)  →  sigmoid
  Output: rating probability ∈ (0, 1)
```

**GraphSAGE intuition:** The embedding of *The Matrix* is updated by aggregating features of all users who rated it — and those users' embeddings are in turn updated by the features of all other movies they watched. By layer 2, a user's embedding reflects a weighted summary of the CV and NLP features of every movie in their history.

### Training Setup

```python
transform = T.RandomLinkSplit(
    num_val=0.1,           # 10% for validation
    num_test=0.1,          # 10% for testing
    disjoint_train_ratio=0.3,   # prevent data leakage
    neg_sampling_ratio=1.0,
)
```

### Ablation Study

| Strategy | Feature Dim | Description |
|----------|-------------|-------------|
| **Baseline** | 64 | Random Gaussian noise |
| **CV-only** | 512 | L2-normalised ResNet-18 embeddings |
| **NLP-only** | 128 | L2-normalised RNN attention embeddings |
| **Multimodal** | 640 | Concatenated CV + NLP, L2-normalised |
| **Pretrained** | 768 | Fine-tuned DistilBERT embeddings |

**Key insights:**
1. **Baseline works** — graph structure alone is a strong signal (collaborative filtering effect)
2. **Content features help** — both CV and NLP improve over random noise
3. **Multimodal wins** — complementary signals from vision and language provide the best recommendation quality
4. **Pre-trained DistilBERT** benefits from large-scale language model knowledge

### Sample Recommendations

```
--- USER 42 HISTORY (action/crime fan) ---
   Die Hard (1988)
   The Matrix (1999)
   Pulp Fiction (1994)
   ...

--- TOP 5 GNN RECOMMENDATIONS (CV features) ---
   Speed (1994)            (0.9231)
   Heat (1995)             (0.9187)
   L.A. Confidential (1997)(0.9044)
   The Usual Suspects      (0.8921)
   Terminator 2 (1991)     (0.8879)
```

---

## Results Summary

| Task | Model | Best Metric |
|------|-------|-------------|
| CV Genre Classification | ResNet-18 fine-tuned | ROC-AUC = **0.712** |
| NLP Genre Classification | RNN + GloVe + Attention | ROC-AUC = **0.605** |
| GNN Link Prediction | Multimodal GraphSAGE | Val AUC > Baseline |

---

## Setup & Installation

### Requirements

```bash
pip install torch torchvision
pip install torch-geometric
pip install scikit-learn pandas numpy matplotlib seaborn
pip install aiohttp tqdm
```

### Running

1. **Download data** — Register for a TMDB API key and update the key in the notebook
2. **Run cells sequentially** — The notebook follows Tasks 1 → 2 → 3
3. **GPU recommended** — Enable GPU in Google Colab (`Runtime → Change runtime type → GPU`)

### Data Files

| File | Description |
|------|-------------|
| `datasets/train_metadata_full.csv` | 15K training movies with posters + plots |
| `datasets/ml_small_metadata_full.csv` | ml-latest-small movies with TMDB data |
| `datasets/cv_embeddings.pt` | ResNet-18 visual embeddings `[N, 512]` |
| `datasets/nlp_embeddings.pt` | RNN attention text embeddings `[N, 128]` |

---

## Repository Structure

```
.
├── DL_LabProject_25_26_dsml00313_part2.ipynb   # Main notebook
├── datasets/
│   ├── ml-latest-small/                        # MovieLens small graph
│   ├── ml-25m/                                 # MovieLens large dataset
│   ├── posters/                                # Downloaded poster images
│   ├── train_metadata_full.csv
│   ├── ml_small_metadata_full.csv
│   ├── cv_embeddings.pt
│   └── nlp_embeddings.pt
├── best_cv_model.pth                           # Saved ResNet-18 weights
├── best_nlp_model.pth                          # Saved RNN weights
├── train_df.pkl / val_df.pkl                   # Cached train/val splits
└── README.md
```

---

**Course:** Deep Learning Laboratory  
**Institution:** National Technical University of Athens (NTUA)  
**Academic Year:** 2025–2026

---

## References

- Harper & Konstan (2015). *The MovieLens Datasets: History and Context*. ACM TIIS.
- Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP.
- Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop.
