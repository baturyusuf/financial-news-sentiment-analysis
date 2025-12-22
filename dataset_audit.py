# dataset_audit.py
import pandas as pd
from src.data import make_dataset

raw = make_dataset.load_raw_data("data/raw/financial_news_market_events_2025.csv")
df = make_dataset.preprocess_data(raw).copy()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Market_Index"]).copy()
df["DateOnly"] = df["Date"].dt.normalize()

print("Rows:", len(df))
print("Unique Market_Index:", df["Market_Index"].nunique())

s = df.groupby(["Market_Index", "DateOnly"]).size()

print("\nRows per (Market_Index, DateOnly) summary:")
print(s.describe())

print("\nHow many (Market_Index, DateOnly) have >1 rows?")
print(int((s > 1).sum()))

print("\nTop 10 most duplicated (Market_Index, DateOnly):")
print(s.sort_values(ascending=False).head(10))
