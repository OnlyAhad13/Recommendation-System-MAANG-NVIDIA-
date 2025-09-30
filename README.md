📚 Enterprise-Scale Recommendation System
🚀 Overview

This project implements an end-to-end recommendation system using a two-tower retrieval model and a Deep & Cross Network (DCN) ranking model.
It follows industry-standard practices from large-scale systems (YouTube, TikTok, Netflix) and is built for both research and production deployment.

✨ Features

Two-Stage Architecture

Retrieval: Multi-tower embeddings with dot-product similarity.

Ranking: Deep & Cross Network (DCN) for feature interactions.

Negative Sampling: Random, Hard, and Mixed strategies.

Feature Support: User & item categorical + numerical features.

Evaluation Metrics: Recall@K, Precision@K, NDCG, MAP, MRR, Coverage, Diversity.

Serving Ready: FAISS-based ANN retrieval + FastAPI + Docker.

📂 Project Structure
.
├── enterprise_recsys.py       # Core system code (retrieval, ranking, metrics, trainer)
├── processed_data.pkl         # Preprocessed dataset (train/val/test splits)
├── requirements.txt           # Dependencies
├── app/                       # FastAPI app for deployment
│   ├── main.py                # FastAPI entrypoint
│   ├── Dockerfile             # Docker config
│   └── requirements.txt       # API dependencies
└── README.md                  # Project documentation

🗂 Dataset

We use MovieLens (100k/1M) or any implicit feedback dataset.

Preprocessing (via DataProcessor):

Convert IDs to strings

Extract time features (hour, day_of_week, is_weekend)

Compute user aggregates (avg rating, count, std)

Compute item aggregates (popularity, avg rating)

Scale numerical features

👉 A preprocessed dataset is included: processed_data.pkl

import pickle

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

train_df, val_df, test_df = data["train"], data["val"], data["test"]

⚙️ Training Pipeline

Retrieval (MultiTower):

User & Item towers produce embeddings in a shared space.

Similarity = dot product.

Loss = sampled softmax with negative sampling.

Ranking (DCN):

Deep & Cross Network learns explicit + nonlinear feature interactions.

Outputs CTR / rating prediction.

Loss = MSE (ratings) + BCE (CTR).

Negative Sampling:

Random → uniform negatives.

Hard → popular unseen items.

Mixed → hybrid.

Multi-task Objective:

L = L_retrieval + α * L_rating + β * L_ctr


Evaluation Metrics:

Retrieval: Recall@K, Precision@K, MAP, MRR, NDCG.

Catalog: Coverage, Diversity.

📊 Metrics
Metric	Purpose
Recall@K	Fraction of relevant items retrieved
Precision@K	Fraction of retrieved items that are relevant
NDCG@K	Rank-sensitive relevance measure
MAP@K	Average precision across ranks
MRR	Rank of first relevant item
Coverage	Fraction of items exposed in recs
Diversity	Item dissimilarity in a rec list

Example formulas (plain text for GitHub):

DCG@k = Σ (2^rel_i − 1) / log2(i+1)

nDCG@k = DCG@k / IDCG@k

MRR = average(1 / rank_of_first_relevant_item)

🧪 Usage
1️⃣ Install Requirements
pip install -r requirements.txt

2️⃣ Train the Model
import enterprise_recsys as recsys
import pickle

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

train_df, val_df, test_df = data["train"], data["val"], data["test"]

config = recsys.ModelConfig(
    embedding_dim=64,
    user_tower_dims=[128, 64],
    item_tower_dims=[128, 64],
    negative_sampling_strategy="mixed",
    ctr_weight=1.0,
    rating_weight=0.5,
    learning_rate=1e-3
)

trainer = recsys.ProductionTrainer(config)
trainer.fit(train_df, val_df)
metrics = trainer.evaluate(test_df)
print(metrics)

3️⃣ Get Recommendations
recs = trainer.recommend(user_id="42", k=10)
print(recs)

🌐 Deployment
FastAPI App (app/main.py)
from fastapi import FastAPI
import pickle, enterprise_recsys as recsys

app = FastAPI()

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)
train_df, val_df, test_df = data["train"], data["val"], data["test"]

config = recsys.ModelConfig(embedding_dim=64)
trainer = recsys.ProductionTrainer(config)
trainer.fit(train_df, val_df)

@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10):
    recs = trainer.recommend(user_id=user_id, k=k)
    return {"user_id": user_id, "recommendations": recs}

Dockerfile (app/Dockerfile)
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

Build & Run
docker build -t recsys-app .
docker run -p 8000:8000 recsys-app

Test API
curl http://localhost:8000/recommend/42?k=10

🛠 Future Work

Add diversity using item embeddings (cosine dissimilarity).

Integrate FAISS/HNSWlib for large-scale retrieval.

Extend with session-based (RNN/Transformer) models.

Deploy to Kubernetes for scaling.

🏆 Summary

This project provides an enterprise-ready recommender system:

Retrieval with MultiTower embeddings

Ranking with DCN

Robust evaluation metrics

Ready for deployment via FastAPI + Docker

Designed to scale to MAANG/NVIDIA-level production workloads.
