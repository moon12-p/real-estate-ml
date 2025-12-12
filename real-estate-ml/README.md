
# Real Estate EDA + ML + Streamlit + MLflow

This repository contains a scaffold for performing EDA, training models (investment classification and price regression),
tracking experiments with MLflow, and deploying an interactive Streamlit dashboard.

**Important**: This ZIP does not include the raw dataset. Place your dataset at `data/raw_real_estate.csv` or the path `/mnt/data/india_housing_prices.csv`.

## Quick start (local)
1. Unzip and open a terminal in the project directory.
2. Create virtual env and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Put your CSV at `data/raw_real_estate.csv` or use the provided path `/mnt/data/india_housing_prices.csv`.
4. Run EDA notebook or scripted EDA:
   ```bash
   python -m src.eda --data data/raw_real_estate.csv --out outputs/
   ```
5. Train models (examples):
   ```bash
   python -m src.train_investment --data data/cleaned_real_estate.csv --target is_good_investment
   python -m src.train_price --data data/cleaned_real_estate.csv --target price
   ```
6. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```
