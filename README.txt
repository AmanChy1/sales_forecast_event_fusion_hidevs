
Weekly Sales Forecasting Project
================================

This project contains two forecasting applications that use the Walmart weekly sales dataset:

1. Streamlit App (app.py)
2. Tkinter Desktop App (sales.py)

Files Included
--------------
- app.py — Streamlit-based interactive web app.
- sales.py — Tkinter GUI desktop application.
- train.csv, features.csv, stores.csv — Required datasets.
- README.txt — Instructions and project overview.

How to Run the Streamlit App
----------------------------
1. Install dependencies:
   pip install streamlit pandas numpy matplotlib statsmodels scikit-learn

2. Keep all CSV files in the same folder as app.py.

3. Run:
   streamlit run app.py

How to Run the Tkinter App
--------------------------
1. Install dependencies:
   pip install pandas numpy matplotlib statsmodels

2. Run:
   python sales.py

Forecasting Model
-----------------
Uses Holt-Winters Exponential Smoothing with:
- Additive trend
- Additive seasonality
- 52-week seasonal period
- Weekly resampled sales data

Requirements
-----------
Python 3.8+ with the listed libraries.
