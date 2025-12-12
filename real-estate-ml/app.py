
import streamlit as st
import pandas as pd
import os
from src.utils import load_and_clean
st.set_page_config(page_title='Real Estate EDA & Models', layout='wide')
st.title('Real Estate EDA & Models — Dashboard')
DATA_DEFAULT = '/mnt/data/india_housing_prices.csv'
upload = st.sidebar.file_uploader('Upload CSV (optional)', type=['csv'])
if upload is not None:
    df = pd.read_csv(upload)
else:
    if os.path.exists(DATA_DEFAULT):
        df = pd.read_csv(DATA_DEFAULT)
    else:
        st.sidebar.info('No dataset found. Upload one to proceed.')
        df = None
tab = st.sidebar.radio('Choose', ['Overview','Price & Size','Location Analysis','Feature Correlation','Investment & Models'])
if df is None:
    st.write('Please upload a dataset to use the app.')
    st.stop()
df = load_and_clean(df if isinstance(df, pd.DataFrame) else 'data/raw_real_estate.csv')
if tab=='Overview':
    st.header('Dataset Overview')
    st.dataframe(df.head())
    st.write('Shape:', df.shape)
    st.write(df.describe(include='all'))
elif tab=='Price & Size':
    st.header('Price & Size Analysis — selected examples')
    st.subheader('Price distribution')
    col = st.selectbox('Choose price column', ['price','price_per_sqft'])
    st.write(df[col].dropna().describe())
    st.hist(df[col].dropna(), bins=50)
    st.pyplot()
else:
    st.write('Other tabs are scaffolds — run the provided scripts for full EDA and model training.')
