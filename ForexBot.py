import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from textblob import TextBlob


# Step 1: Data Collection
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
SYMBOL = "USDEUR"
INTERVAL = "15min"

response = requests.get(
    f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={SYMBOL}&interval={INTERVAL}&apikey={API_KEY}"
)
data = response.json()
time_series_data = data["Time Series (15min)"]


def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(close_prices, short_window=12, long_window=26, signal_window=9):
    short_ema = close_prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram


# Step 2: Data Preprocessing and Feature Engineering
time_series_data = data["Time Series (15min)"]

# Convert data into a DataFrame
df = pd.DataFrame.from_dict(time_series_data, orient="index")
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Convert data types and select relevant columns
df["1. open"] = pd.to_numeric(df["1. open"])
df["2. high"] = pd.to_numeric(df["2. high"])
df["3. low"] = pd.to_numeric(df["3. low"])
df["4. close"] = pd.to_numeric(df["4. close"])
df["5. volume"] = pd.to_numeric(df["5. volume"])

# Calculate technical indicators
df["15min_MA"] = df["4. close"].rolling(window=3).mean()
df["30min_MA"] = df["4. close"].rolling(window=6).mean()
df["15min_RSI"] = calculate_rsi(df["4. close"], window=14)
df["30min_RSI"] = calculate_rsi(df["4. close"], window=30)
df["15min_MACD"] = calculate_macd(
    df["4. close"], short_window=12, long_window=26, signal_window=9
)


def fetch_fundamental_data(symbol, start_date, end_date):
    # Fetch fundamental data using yfinance
    stock = yf.Ticker(symbol)
    fundamental_data = stock.history(start=start_date, end=end_date)

    # Additional processing or calculations
    fundamental_data["daily_return"] = fundamental_data["Close"].pct_change()
    fundamental_data["moving_avg_5"] = (
        fundamental_data["Close"].rolling(window=5).mean()
    )
    fundamental_data["moving_avg_20"] = (
        fundamental_data["Close"].rolling(window=20).mean()
    )

    # You can add more processing steps based on your strategy's needs

    return fundamental_data


# Integrate fundamental data (example: Non-farm Payrolls)
fundamental_data = (
    fetch_fundamental_data()
)  # Implement a function to fetch fundamental data
df = df.merge(fundamental_data, left_index=True, right_index=True)


def fetch_sentiment_data(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    return sentiment_score


# Perform sentiment analysis (example: Sentiment scores from news articles)
sentiment_data = fetch_sentiment_data()  # Implement a function to fetch sentiment data
df = df.merge(sentiment_data, left_index=True, right_index=True)


# Step 3: Labeling and Target Creation (Moving Average Crossover Strategy)
df["15min_MA"] = df["4. close"].rolling(window=3).mean()
df["30min_MA"] = df["4. close"].rolling(window=6).mean()

df["position"] = 0  # 0: No position, 1: Long, -1: Short
df.loc[df["15min_MA"] > df["30min_MA"], "position"] = 1
df.loc[df["15min_MA"] < df["30min_MA"], "position"] = -1

# Create target labels for classification (shifted by one step)
df["target"] = df["position"].shift(-1)
df.dropna(subset=["target"], inplace=True)

# Step 4: Split Data and Model Training
X = df.drop(columns=["position", "target"])  # Features for prediction
y = df["target"]

# Step 5: Model Training
model = LogisticRegression()
model.fit(X, y)

# Step 6: Backtesting and Performance Evaluation
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"Backtesting Accuracy: {accuracy}")

# Step 7: Execution and Risk Management
# ... (implement an execution strategy based on predictions, considering transaction costs and risk management) ...

# Step 8: Continuous Monitoring and Updating
# ... (regularly assess and update your strategy as needed) ...
