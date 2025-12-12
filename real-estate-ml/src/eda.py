
# Simple scripted EDA that writes a few summary CSVs and plots.
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_and_clean, detect_outliers_iqr
sns.set()

def run(data_path, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_clean(data_path)
    # 1. Price distribution
    df['price'].to_csv(os.path.join(out_dir, 'price_series.csv'), index=False)
    plt.figure(figsize=(8,5))
    sns.histplot(df['price'].dropna(), bins=50, kde=True)
    plt.title('Price distribution')
    plt.savefig(os.path.join(out_dir, 'price_distribution.png'), dpi=150)
    plt.close()
    # 2. Size distribution
    if 'size_sqft' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['size_sqft'].dropna(), bins=50, kde=True)
        plt.title('Size (sqft) distribution')
        plt.savefig(os.path.join(out_dir, 'size_distribution.png'), dpi=150)
        plt.close()
    # 3. price per sqft by property_type
    if 'property_type' in df.columns and 'price_per_sqft' in df.columns:
        grp = df.groupby('property_type')['price_per_sqft'].median().sort_values(ascending=False)
        grp.to_csv(os.path.join(out_dir, 'price_per_sqft_by_property_type.csv'))
    # 4. size vs price scatter
    if 'size_sqft' in df.columns and 'price' in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='size_sqft', y='price', alpha=0.4)
        sns.regplot(data=df, x='size_sqft', y='price', scatter=False, lowess=True, color='red')
        plt.savefig(os.path.join(out_dir, 'size_vs_price.png'), dpi=150)
        plt.close()
    # 5. Outliers
    outliers_pps = detect_outliers_iqr(df['price_per_sqft']) if 'price_per_sqft' in df.columns else pd.Series([])
    outliers_pps.to_csv(os.path.join(out_dir, 'outliers_price_per_sqft.csv'), index=False)
    print('Outputs written to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    run(args.data, args.out)
