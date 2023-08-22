import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Data Collection
# API_KEY = "RVPVYU9K17NXJ41L"
# SYMBOL = "USDEUR"
# INTERVAL = "15min"

response = requests.get(
    f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=USDEUR&interval=1min&apikey=RVPVYU9K17NXJ41L&outputsize=full&startdate=2023-06-21&enddate=2023-08-21"
)
data = response.json()
time_series_data = data["Time Series (1min)"]

# Step 2: Data Preprocessing and Feature Engineering
df = pd.DataFrame.from_dict(time_series_data, orient="index")
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df["4. close"] = pd.to_numeric(df["4. close"])

# Calculate technical indicators (example: Moving Averages)
df["15min_MA"] = df["4. close"].rolling(window=3).mean()
df["30min_MA"] = df["4. close"].rolling(window=6).mean()

# ... (Add more technical indicators or feature engineering steps)

# Step 3: Labeling and Target Creation (Moving Average Crossover Strategy)
df["position"] = 0  # 0: No position, 1: Long, -1: Short
df.loc[df["15min_MA"] > df["30min_MA"], "position"] = 1
df.loc[df["15min_MA"] < df["30min_MA"], "position"] = -1

# Create target labels for classification (shifted by one step)
df["target"] = df["position"].shift(-1)
df.dropna(subset=["target"], inplace=True)

# Split Data for Training
X = df.drop(columns=["position", "target"])  # Features for prediction
y = df["target"]

## Step 4: Model Training
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

model = LogisticRegression()
model.fit(X_imputed, y)


# Step 5: Save the Trained Model
joblib.dump(model, "forex_model.joblib")
