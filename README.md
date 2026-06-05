# Steam Game Recommender System

A collaborative filtering recommender system built using **Apache Spark MLlib** on the Steam 200k dataset. The system predicts game recommendations for users based on their play history using **ALS (Alternating Least Squares) matrix factorisation**.

**MSc Data Science — University of Salford, Manchester**

---

## Overview

Steam is an online game distribution platform with millions of users. This project builds a personalised game recommender system using implicit feedback (specifically, hours played) to predict which games a user is likely to engage with.

The full pipeline covers data preprocessing, exploratory data analysis, model training, hyperparameter tuning with MLflow experiment tracking, and recommendation generation.

---

## Dataset

**Steam 200k dataset** (available on [Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games))

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `name_of_game` | Game title |
| `behaviour` | Either `purchase` or `play` |
| `value` | Hours played (for play) or 1.0 (for purchase) |

- **12,393** users with purchase records
- **11,350** users with play records
- **3,600** unique games
- Matrix sparsity: **99.83%**

---

## Approach

### Preprocessing
- Only `play` interactions used as playtime is a stronger signal of user preference than purchase alone
- 1,043 users (approximately 8%) with purchase-only records excluded, a small loss justified by stronger signal quality
- Log transformation (`log1p`) applied to hours played to compress right-skewed distribution (mean: 48.88 hours, median: 4.5 hours)
- `StringIndexer` used to convert game names and user IDs to integer indices required by ALS

### Exploratory Data Analysis
- **Dota 2** is the most popular game with 4,841 players and 981,684 total hours played
- Median user has played just **1 game**; 75% of users have played 3 or fewer
- Dataset is highly sparse with a small number of power users skewing the distribution

### Model Training
- **Algorithm:** ALS (Alternating Least Squares) collaborative filtering via Spark MLlib
- **Train/test split:** 80/20 with fixed seed (42) for reproducibility
- **Cold start strategy:** `drop` to remove users/items with no training data from evaluation
- All experiments tracked with **MLflow**

### Hyperparameter Tuning

Grid search over 9 combinations:

| Hyperparameter | Values tested |
|----------------|--------------|
| `rank` | 10, 20, 30 |
| `regParam` | 0.01, 0.1, 1.0 |
| `maxIter` | 10 (fixed) |

**Best model:** `rank=30`, `regParam=0.1` → **RMSE: 1.4620** (vs baseline 1.5054)

Key finding: `regParam=0.01` consistently underperformed as too little regularisation causes overfitting on sparse data.

---

## Results

- Best ALS model achieved **RMSE of 1.4620**
- Top 10 game recommendations generated for all users
- Recommendations are personalised: different user profiles receive meaningfully different game suggestions (e.g. User 0 recommended strategy games, User 1 receives a different genre set)
- Integer game IDs mapped back to actual game names for interpretability

---

## Tech Stack

- **Platform:** Databricks
- **Processing:** Apache Spark (PySpark)
- **ML Library:** MLlib (ALS)
- **Experiment Tracking:** MLflow
- **Language:** Python

---

## Files

```
steam-game-recommender/
│
├── README.md
├── 00785442_Assignment_2_Task_2.ipynb    # Jupyter notebook (full pipeline)
└── 00785442_Assignment_2_Task_2.html     # Exported notebook with outputs
```

## How to Run

1. Upload the notebook to your Databricks workspace
2. Attach to a cluster with Spark and MLlib available
3. Update the dataset path to point to your copy of `steam-200k.csv`:
   ```python
   df = spark.read.csv("/your/path/to/steam-200k.csv", header=False, inferSchema=True)
   ```
4. Run all cells in order

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games).
