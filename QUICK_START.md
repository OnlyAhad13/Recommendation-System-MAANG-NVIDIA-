# 🚀 Quick Start Guide

Get up and running with the restructured recommendation system in 3 steps!

---

## ⚡ TL;DR - 3 Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data (if not done already)
python scripts/preprocess.py --data_dir data/raw --output data/processed/processed_data.pkl

# 3. Train model
python scripts/train.py --data data/processed/processed_data.pkl --output_dir outputs/models/experiment_001 --epochs 5
```

---

## 📋 Step-by-Step

### 1. Setup Environment

```bash
cd "/home/onlyahad/Desktop/Recommendation System"

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data

```bash
# Check if data is in the right place
ls data/raw/
# Should see: movies.dat, ratings.dat, users.dat, README

# If processed data doesn't exist, create it:
python scripts/preprocess.py \
    --data_dir data/raw \
    --output data/processed/processed_data.pkl
```

### 3. Train Model

```bash
# Basic training (5 epochs, ~5 minutes on CPU)
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/quick_test \
    --epochs 5

# View results
ls outputs/models/quick_test/
# You'll see: best_model.keras, encoder.keras, faiss.idx, metrics.json, training_log.csv
```

---

## 🔍 Verify Installation

Test that everything works:

```python
# Run Python interpreter
python3

# Try importing the package
>>> from src.config import ModelConfig
>>> from src.trainer import ProductionTrainer
>>> config = ModelConfig(embedding_dim=64, batch_size=2048, epochs_retrieval=1)
>>> print(config)
>>> exit()
```

If no errors, you're good to go! ✅

---

## 📊 View Training Results

### Option 1: Check Metrics File
```bash
cat outputs/models/quick_test/metrics.json
```

### Option 2: View with TensorBoard
```bash
tensorboard --logdir outputs/models/quick_test/logs
# Open browser to http://localhost:6006
```

### Option 3: Check CSV Log
```bash
head outputs/models/quick_test/training_log.csv
```

---

## 🎯 Common Tasks

### Test Different Configurations

```bash
# Larger model
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/large_model \
    --embedding_dim 256 \
    --batch_size 8192 \
    --epochs 10

# Different negative sampling
python scripts/train.py \
    --data data/processed/processed_data.pkl \
    --output_dir outputs/models/hard_negatives \
    --negative_sampling hard \
    --num_hard_negatives 20
```

### Use in Python Script

```python
from src.config import ModelConfig
from src.trainer import ProductionTrainer

# Create config
config = ModelConfig(
    embedding_dim=128,
    batch_size=4096,
    epochs_retrieval=5
)

# Train
trainer = ProductionTrainer(config, 'outputs/models/my_experiment')
model, history = trainer.train('data/processed/processed_data.pkl')

# Check metrics
print(f"Final validation loss: {history.history['val_loss'][-1]}")
```

---

## 🧹 Cleanup Old Files (Optional)

After verifying the new structure works:

```bash
# Remove old output directories
rm -rf main_files/enterprise_recsys_output
rm -rf main_files/out_test
rm -rf main_files/output
rm -rf main_files/recsys_output

# Remove empty directory
rmdir movies_dataset

# See MIGRATION.md for complete cleanup instructions
```

---

## 📖 Need More Help?

- **Full Documentation**: See `README.md`
- **Migration Guide**: See `MIGRATION.md`
- **Summary of Changes**: See `RESTRUCTURING_SUMMARY.md`
- **All Training Options**: Run `python scripts/train.py --help`

---

## 🎉 You're Ready!

The project is now properly structured and ready for:
- ✅ Development
- ✅ Experimentation
- ✅ Production deployment
- ✅ Team collaboration

Happy coding! 🚀

