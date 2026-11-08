# 💻 Laptop Price Predictor

A Streamlit web application that predicts laptop retail prices in Indian Rupees (₹) based on brand specifications and hardware configurations using a trained Random Forest Regressor pipeline[cite: 1, 4].

---

## 📌 Features

- **Real-Time Prediction:** Instant price estimation based on custom hardware specs[cite: 1, 4].
- **Dynamic PPI Calculator:** Computes display pixel density on the fly from resolution and screen size[cite: 1, 4].
- **Preprocessing Pipeline:** End-to-end Scikit-Learn `Pipeline` handling one-hot encoding, scaling, and inference without data leakage.

---

## 📊 Model Performance

Evaluated on an unseen 20% holdout test dataset (261 records):

| Metric | Score |
| :--- | :--- |
| **Model** | Random Forest Regressor |
| **$R^2$ Score** | **0.8781** |
| **Mean Absolute Error (MAE)** | **₹7,971.24** |
| **Root Mean Squared Error (RMSE)** | **₹11,360.02** |

---

## 📁 Project Structure

```text
Laptop-Price-Prediction-New/
├── data/
│   └── laptop_data.csv             # Raw dataset (1,303 records)
├── models/
│   └── laptop_price_model.joblib   # Serialized model pipeline
├── notebooks/
│   ├── 01_EDA.ipynb                # Data exploration and visual analysis
│   └── 02_Model_Development.ipynb  # Preprocessing, tuning, and evaluation
├── app.py                          # Streamlit web application
├── requirements.txt                # Project dependencies
└── README.md