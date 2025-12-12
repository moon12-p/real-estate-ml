
# Train a simple regression model for price prediction (sketch)
import argparse
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .utils import load_and_clean

def run(data_path, out_model='model_price.joblib'):
    df = load_and_clean(data_path)
    features = ['size_sqft','price_per_sqft','amenities_count','parking_flag','age']
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features+['price'])
    X = df[features]; y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print('RMSE:', rmse, 'R2:', r2)
    joblib.dump(model, out_model)
    try:
        with mlflow.start_run():
            mlflow.log_metric('rmse', float(rmse))
            mlflow.log_metric('r2', float(r2))
            mlflow.sklearn.log_model(model, 'price_model')
    except Exception as e:
        print('MLflow logging skipped:', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model_price.joblib')
    args = parser.parse_args()
    run(args.data, args.out)
