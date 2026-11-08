# 💻 Laptop Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://laptop-price-prediction-01.streamlit.app/)

> **Live Demo:** [laptop-price-prediction-01.streamlit.app](https://laptop-price-prediction-01.streamlit.app/)

A clean, end-to-end Machine Learning web application that predicts laptop retail prices in Indian Rupees (₹) based on brand and hardware specifications using a trained Random Forest Regressor pipeline.

---

## 📌 Features

- **Live Real-Time Inference:** Instant price prediction powered by an end-to-end Scikit-Learn pipeline.
- **Dynamic PPI Calculation:** Automatically computes pixel density based on selected screen dimensions and resolution.
- **Zero Data Leakage:** Preprocessing transformations (scaling and one-hot encoding) are encapsulated within a `ColumnTransformer` fitted strictly on training data.

---

## 📊 Model Performance

Evaluated on an unseen 20% holdout test dataset (261 records):

| Metric | Score |
| :--- | :--- |
| **Final Model** | Random Forest Regressor |
| **R² Score** | **0.8782** |
| **Mean Absolute Error (MAE)** | **₹7,971.24** |
| **Root Mean Squared Error (RMSE)** | **₹11,360.02** |

---

## 📁 Project Structure

```text
Laptop-Price-Prediction-New/
├── data/
│   └── laptop_data.csv             # Dataset (1,303 records)
├── models/
│   └── laptop_price_model.joblib   # Serialized ML pipeline
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis & visual insights
│   └── 02_Model_Development.ipynb  # Pipeline creation, tuning, & evaluation
├── app.py                          # Streamlit web application
├── requirements.txt                # Project dependencies
├── .gitignore                      # Ignored caches & checkpoints
└── README.md                       # Project documentation
