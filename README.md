# 🏠 Airbnb Price Predictor

Predicting Airbnb listing prices using location and property characteristics, with a comparison of three regression models: **Linear Regression**, **Random Forest**, and **LightGBM**.

---

## 📌 Project Overview

Given a set of Airbnb listing features (location, property type, room type, host attributes, reviews), the goal is to predict the **nightly price** of a listing.

This is a **regression problem**. Prices are log-transformed (`log1p`) before training to handle skewness, and converted back (`expm1`) for interpretation.

---

## 📂 Project Structure

```
airbnb-price-predictor/
│
├── data/
│   └── raw/
│       └── Listings.csv          # Raw dataset
│
├── src/
│   ├── load_data.py              # Data loading
│   └── model.py                  # Preprocessing, training, evaluation
│
├── notebooks/
│   └── airbnb_price_analysis.ipynb  # Step-by-step walkthrough
│
├── main.py                       # Entry point
├── requirements.txt
└── README.md
```

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `latitude`, `longitude` | Geographic coordinates |
| `neighbourhood`, `district`, `city` | Location categories |
| `room_type`, `property_type` | Type of space |
| `accommodates`, `bedrooms` | Size of the listing |
| `minimum_nights` | Booking restriction |
| `host_is_superhost` | Host quality flag |
| `review_scores_rating`, `review_scores_location` | Guest review scores |
| `instant_bookable` | Booking convenience flag |

---

## 🤖 Models

| Model | Description |
|---|---|
| Linear Regression | Baseline model, interpretable |
| Random Forest | Ensemble of decision trees, handles non-linearity |
| LightGBM | Gradient boosting, fast and high-performance |

All models are evaluated on:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **10-fold Cross-Validation R²**

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Goncalo-math/AirBnB-Listings.git
cd airbnb-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Place `Listings.csv` inside `data/raw/`.

**4. Run the pipeline**
```bash
python main.py
```

**5. Or explore the notebook**
```bash
jupyter notebook notebooks/airbnb_price_analysis.ipynb
```

---

## 📊 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~0.45 | ~0.60 | ~0.45 |
| Random Forest | ~0.35 | ~0.48 | ~0.58 |
| LightGBM | ~0.30 | ~0.42 | ~0.63 |

> Results are on log-transformed prices. LightGBM outperforms the other models.

---

## 🧰 Tech Stack

- Python 3.12
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📄 Dataset

The dataset is based on publicly available [Airbnb listing data](https://www.kaggle.com/datasets/ulrikthygepedersen/airbnb-listings). It contains property listings with pricing, host information, location, and review scores.
