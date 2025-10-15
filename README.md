# 🎬 Enterprise-Scale Recommendation System

A production-ready recommendation system using **two-tower retrieval** and **Deep & Cross Network (DCN)** ranking models. Built following industry best practices from systems like YouTube, Netflix, and TikTok.

---

## 🚀 Features

### Two-Stage Architecture
- **Retrieval Stage**: Multi-tower embeddings with dot-product similarity
- **Ranking Stage**: Deep & Cross Network (DCN) for feature interactions

### Advanced Capabilities
- ✅ Multiple negative sampling strategies (Random, Hard, Mixed)
- ✅ Rich feature engineering (user/item statistics, temporal features)
- ✅ Comprehensive evaluation metrics (Recall@K, Precision@K, NDCG, MAP, MRR)
- ✅ FAISS-based approximate nearest neighbor search
- ✅ Multi-task learning (CTR + Rating prediction)
- ✅ Class weights for handling imbalanced datasets
- ✅ FastAPI REST API with Docker support
- ✅ Modular, production-ready codebase
- ✅ W&B integration for experiment tracking

---

## 📂 Project Structure

```
Recommendation System/
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── QUICK_START.md              # Quick start guide
│
├── data/                        # Data directory
│   ├── raw/                     # Raw MovieLens dataset
│   │   ├── movies.dat
│   │   ├── ratings.dat
│   │   ├── users.dat
│   │   └── README
│   └── processed/               # Preprocessed data (gitignored)
│       └── processed_data.pkl
│
├── src/                         # Source code package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Model configuration
│   ├── models.py                # Model architectures (DCN, MultiTower)
│   ├── data_processing.py       # Data loading and feature engineering
│   ├── evaluation.py            # Metrics and callbacks
│   ├── trainer.py               # Training orchestration
│   └── preprocessing.py         # Dataset preprocessing pipeline
│
├── app/                         # FastAPI application
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI main application
│   ├── recommendation_service.py # Recommendation service
│   ├── simple_model_loader.py   # Simple model loader
│   ├── model_service.py         # Model service
│   ├── start_api.sh             # API startup script
│   ├── test_api.py              # API testing script
│   ├── docker-compose.yml       # Docker configuration
│   ├── Dockerfile               # Docker image
│   └── requirements.txt         # API dependencies
│
├── scripts/                     # Executable scripts
│   ├── train.py                 # Training entrypoint
│   └── preprocess.py            # Data preprocessing script
│
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_basic_analysis.ipynb
│
├── outputs/                     # Training outputs (gitignored)
│   ├── models/                  # Saved models
│   │   └── experiment_001/      # Experiment outputs
│   ├── logs/                    # Training logs
│   └── metrics/                 # Evaluation results
│
└── configs/                     # Configuration files
```

---

## 🗂 Dataset

This project uses the **MovieLens** dataset (1M ratings). The dataset includes:
- User ratings for movies
- Movie metadata (title, genres)
- User demographics (age, gender, occupation)

### Data Splits
- **Training**: 80% (temporal)
- **Validation**: 10% (temporal)
- **Testing**: 10% (temporal)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd recommendation-system
```

### 2. Install Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. (Optional) Install GPU Support
```bash
# For FAISS GPU support
pip install faiss-gpu

# Verify TensorFlow GPU
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

---

## 🚀 Quick Start

### Step 1: Preprocess the Data

```bash
python scripts/preprocess.py \
    --data_dir data/raw \
    --output data/processed/processed_data.pkl
```

### Step 2: Train the Model

```bash
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/experiment_001 \
    --embedding_dim 128 \
    --batch_size 4096 \
    --epochs 5 \
    --learning_rate 0.01 \
    --negative_sampling mixed
```

### Step 3: Start the API Server

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations for a user
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1001, "num_recommendations": 10}'
```

### Step 5: View Results

Training artifacts will be saved to `outputs/models/experiment_001/`:
- `best_model.keras` - Best model checkpoint
- `encoder.keras` - Encoder model for inference
- `faiss.idx` - FAISS index for fast retrieval
- `training_log.csv` - Training metrics per epoch
- `metrics.json` - Final evaluation metrics
- `logs/` - TensorBoard logs

View with TensorBoard:
```bash
tensorboard --logdir outputs/models/experiment_001/logs
```

---

## 🌐 API Documentation

### FastAPI Endpoints

The recommendation system includes a FastAPI server with the following endpoints:

#### Health Check
```http
GET /health
```
Returns the health status of the API.

#### Get Recommendations
```http
POST /recommendations
Content-Type: application/json

{
    "user_id": 1,
    "num_recommendations": 10
}
```

**Response:**
```json
{
    "user_id": 1,
    "recommendations": [
        {"item_id": 123, "score": 0.95},
        {"item_id": 456, "score": 0.89}
    ],
    "num_recommendations": 10
}
```

#### Get Similar Items
```http
POST /similar_items
Content-Type: application/json

{
    "item_id": 123,
    "num_similar": 5
}
```

### Running the API

#### Development Mode
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Mode
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using Docker
```bash
cd app
docker-compose up --build
```

### API Testing

Test the API endpoints:
```bash
# Run the test script
python app/test_api.py

# Or test manually
curl http://localhost:8000/health
```

---

## 📊 Model Architecture

### Retrieval Model (Two-Tower)
```
User Tower:                    Item Tower:
  User ID Embedding              Item ID Embedding
  + Feature Embeddings           + Feature Embeddings
       ↓                              ↓
  Dense(256) + BN + Dropout      Dense(256) + BN + Dropout
       ↓                              ↓
  Dense(128) + BN + Dropout      Dense(128) + BN + Dropout
       ↓                              ↓
  Dense(128)                     Dense(128)
       ↓                              ↓
  User Embedding ────────────→ Dot Product Similarity
```

### Ranking Model (DCN)
```
[User Emb || Item Emb]
       ↓
Deep & Cross Network
  ├─ Cross Layers (3x)
  └─ Deep Layers [256, 128, 64]
       ↓
    Concat
       ↓
  ├─ Rating Head (MSE)
  └─ CTR Head (BCE)
```

### Multi-Task Loss
```
L_total = L_retrieval + α * L_rating + β * L_ctr
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of relevant items retrieved in top-K |
| **Precision@K** | Fraction of top-K items that are relevant |
| **NDCG@K** | Normalized Discounted Cumulative Gain (rank-aware) |
| **MAP@K** | Mean Average Precision across all ranks |
| **MRR** | Mean Reciprocal Rank of first relevant item |
| **Coverage** | Fraction of catalog items recommended |
| **Diversity** | Uniqueness of items in recommendation lists |

---

## 🔧 Configuration

### Key Hyperparameters

Edit `src/config.py` or pass arguments to `scripts/train.py`:

```python
ModelConfig(
    # Architecture
    embedding_dim=128,              # Embedding dimension
    user_tower_dims=[256, 128],     # User tower architecture
    item_tower_dims=[256, 128],     # Item tower architecture
    cross_layers=3,                 # DCN cross layers
    dnn_dims=[256, 128, 64],        # DCN deep layers
    
    # Training
    batch_size=2048,
    learning_rate_retrieval=0.01,
    epochs_retrieval=5,
    
    # Negative Sampling
    negative_sampling_strategy="mixed",  # random, hard, mixed
    num_hard_negatives=5,
    num_random_negatives=50,
    
    # Multi-task weights
    ctr_weight=0.3,
    rating_weight=0.7,
    
    # Class weights for imbalanced datasets
    use_class_weights=True,  # Automatically calculated for balanced training
)
```

---

## 🎯 Advanced Usage

### Custom Training with W&B

```bash
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/experiment_001 \
    --use_wandb \
    --embedding_dim 128 \
    --epochs 20
```

### Distributed Training (Multi-GPU)

```bash
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/distributed_run \
    --distributed_strategy mirrored \
    --batch_size 8192
```

### Hyperparameter Tuning

```bash
# Example: Test different negative sampling strategies
for strategy in random hard mixed; do
    python scripts/train.py \
        --data data/processed/processed_data.pkl \
        --output_dir outputs/models/neg_sampling_${strategy} \
        --negative_sampling ${strategy}
done
```

---

## 📝 Code Organization

### Source Modules

- **`src/config.py`**: Configuration dataclass with all hyperparameters
- **`src/models.py`**: Model architectures (DeepCrossNetwork, MultiTowerModel, MultiTaskModel)
- **`src/data_processing.py`**: Data loading, feature engineering, negative sampling
- **`src/evaluation.py`**: Evaluation metrics and custom callbacks
- **`src/trainer.py`**: Training orchestration and artifact management
- **`src/preprocessing.py`**: MovieLens data preprocessing pipeline

### Scripts

- **`scripts/train.py`**: Main training entrypoint with CLI arguments

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train.py --batch_size 2048 ...

# Use mixed precision (automatic with GPU)
# Already enabled when GPU is detected
```

### Slow Training
```bash
# Increase batch size (if you have memory)
python scripts/train.py --batch_size 8192 ...

# Reduce model complexity
python scripts/train.py --embedding_dim 64 --cross_layers 2 ...
```

### Import Errors
```bash
# Make sure you're in the project root
cd /path/to/recommendation-system

# Install in development mode (optional)
pip install -e .
```

---

## 🛠 Future Enhancements

- [x] Add REST API with FastAPI
- [x] Add Docker containerization
- [ ] Implement online learning capabilities
- [ ] Add diversity constraints in ranking
- [ ] Session-based recommendations (RNN/Transformer)
- [ ] A/B testing framework
- [ ] Deployment to Kubernetes
- [ ] Add model versioning and A/B testing
- [ ] Implement real-time feature serving
- [ ] Add monitoring and alerting

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References

- [Deep & Cross Network (DCN)](https://arxiv.org/abs/1708.05123)
- [Two-Tower Models for Retrieval](https://research.google/pubs/pub48840/)
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
