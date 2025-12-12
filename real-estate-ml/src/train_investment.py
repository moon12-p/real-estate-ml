
# Train a simple investment classification model and log with MLflow (sketch).
import argparse
import mlflow, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from .utils import load_and_clean

def create_target(df):
    # Example heuristic: top 25% price_per_sqft and available immediately
    df = df.copy()
    df['is_good_investment'] = 0
    if 'price_per_sqft' in df.columns:
        thresh = df['price_per_sqft'].quantile(0.75)
        df.loc[(df['price_per_sqft']>=thresh) & (df.get('availability','').astype(str).str.contains('ready|immediate', case=False, na=False)), 'is_good_investment'] = 1
    return df

def run(data_path, out_model='model_investment.joblib'):
    df = load_and_clean(data_path)
    df = create_target(df)
    features = ['price_per_sqft','size_sqft','amenities_count','parking_flag','age']
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features+['is_good_investment'])
    X = df[features]; y = df['is_good_investment']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds)
    print('Accuracy:', acc, 'F1:', f1)
    joblib.dump(clf, out_model)
    # MLflow logging (optional)
    try:
        with mlflow.start_run():
            mlflow.log_metric('accuracy', float(acc))
            mlflow.log_metric('f1', float(f1))
            mlflow.sklearn.log_model(clf, 'investment_model')
    except Exception as e:
        print('MLflow logging skipped:', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model_investment.joblib')
    args = parser.parse_args()
    run(args.data, args.out)
