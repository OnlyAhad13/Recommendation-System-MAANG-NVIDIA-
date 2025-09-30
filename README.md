📚 Recommendation System — Enterprise-Scale (Retrieval + Ranking)
🚀 Project Overview

This project implements an end-to-end recommendation system following the two-tower architecture for retrieval and Deep & Cross Networks (DCN) for ranking.
It is designed to scale from academic datasets (MovieLens) to production-grade recommendation tasks.

Key Features

Two-Stage Design:

Retrieval: Efficient candidate generation using user/item embeddings.

Ranking: Precise ordering of candidates with feature interactions (DCN).

Advanced Negative Sampling: Mixed strategy with random + hard negatives.

Rich Feature Support: User & item categorical and numerical features.

Evaluation Metrics: Recall@K, Precision@K, NDCG, MAP, MRR, Coverage, Diversity.

Serving Ready: FAISS-based ANN retrieval for production, deployable with FastAPI & Docker.

📂 Project Structure
.
├── enterprise_recsys.py      # Core system code
├── processed_data.pkl        # Preprocessed dataset (train/val/test splits)
├── requirements.txt          # Dependencies
├── app/                      # FastAPI serving app (to be added)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
└── README.md                 # This documentation

🗂 Dataset

We use MovieLens 100k/1M as the primary dataset (or any implicit feedback dataset).
Preprocessing (via DataProcessor) includes:

Renaming IDs & converting to string

Extracting temporal features (hour, day_of_week, weekend)

User stats: avg rating, rating std, count

Item stats: popularity, avg rating

Scaling numerical features

👉 In this repo, we provide processed_data.pkl, which contains:

train → Training DataFrame

val → Validation DataFrame

test → Test DataFrame

You can load it directly:

import pickle
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

train_df, val_df, test_df = data["train"], data["val"], data["test"]

⚙️ Training Pipeline

Retrieval Model (MultiTower):

User & Item towers → Embeddings in shared space.

Similarity = dot product.

Loss = retrieval loss with negative sampling.

Ranking Model (DCN):

Deep & Cross Network for explicit/nonlinear feature interactions.

Predicts CTR/rating.

Loss = combination of regression (MSE) and classification (BCE).

Negative Sampling:

Random: uniform selection.

Hard: most popular unseen items.

Mixed: hybrid for better gradient signal.

Multi-task Learning:
Total loss:

𝐿
=
𝐿
𝑟
𝑒
𝑡
𝑟
𝑖
𝑒
𝑣
𝑎
𝑙
+
𝛼
𝐿
𝑟
𝑎
𝑡
𝑖
𝑛
𝑔
+
𝛽
𝐿
𝑐
𝑡
𝑟
L=L
retrieval
	​

+αL
rating
	​

+βL
ctr
	​


Evaluation:

Retrieval Metrics: Recall@K, Precision@K, MRR, MAP, NDCG.

Catalog Metrics: Coverage, Diversity.

📊 Metrics
Metric	Purpose
Recall@K	Fraction of relevant items retrieved
Precision@K	Fraction of retrieved items that are relevant
NDCG@K	Rank-sensitive relevance measure
MAP@K	Average precision across ranks
MRR	Rank of first relevant item
Coverage	Fraction of items exposed in recs
Diversity	Item dissimilarity in a rec list
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

🌐 Deployment with FastAPI + Docker

FastAPI App (app/main.py):

from fastapi import FastAPI
import pickle, enterprise_recsys as recsys

app = FastAPI()

# Load model
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


Dockerfile (app/Dockerfile):

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


Build & Run:

docker build -t recsys-app .
docker run -p 8000:8000 recsys-app


Test API:

curl http://localhost:8000/recommend/42?k=10

🛠 Future Work

Implement feature-based diversity with item embeddings.

Integrate ANN search with FAISS/HNSWlib for large-scale serving.

Add session-based models (RNN/Transformer) for sequential recs.

Deploy to Kubernetes for auto-scaling.

🏆 Summary

This project implements an enterprise-grade recommender system:

Retrieval with MultiTower embeddings.

Ranking with Deep & Cross Networks.

Rich evaluation metrics.

Ready for production deployment with FastAPI & Docker.

It’s designed not just for research but to scale to MAANG/NVIDIA-level production use cases.
