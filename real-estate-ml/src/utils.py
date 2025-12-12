
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    # standardize names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # convert price and size
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'size_sqft' in df.columns:
        df['size_sqft'] = pd.to_numeric(df['size_sqft'], errors='coerce')
    # price_per_sqft
    if 'price' in df.columns and 'size_sqft' in df.columns:
        df['price_per_sqft'] = df['price'] / df['size_sqft']
    # age
    if 'year_built' in df.columns:
        df['age'] = 2025 - pd.to_numeric(df['year_built'], errors='coerce')
    else:
        df['age'] = np.nan
    # amenities count
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].fillna('').apply(lambda x: 0 if str(x).strip()=='' else len(str(x).split(',')))
    # parking flag
    if 'parking' in df.columns:
        df['parking_flag'] = df['parking'].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['no','0','none'] else 1)
    # impute numeric with median
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    # categorical fill
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna('unknown')
    return df

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]
