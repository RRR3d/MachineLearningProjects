import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import json

# Load the trained model
model = joblib.load("forex_model.joblib")

# Load and preprocess test data from JSON
with open("query.json", "r") as json_file:
    data = json.load(json_file)

# Extract the "Time Series (1min)" data
time_series_data = data["Time Series (1min)"]

# Convert the JSON data to a DataFrame
df_test = pd.DataFrame.from_dict(time_series_data, orient="index")
df_test.index = pd.to_datetime(df_test.index)
df_test["4. close"] = pd.to_numeric(df_test["4. close"])

# Calculate technical indicators (example: Moving Averages)
df_test["15min_MA"] = df_test["4. close"].rolling(window=3).mean()
df_test["30min_MA"] = df_test["4. close"].rolling(window=6).mean()

# Simulate position based on moving average crossover strategy
df_test["position"] = 0
df_test.loc[df_test["15min_MA"] > df_test["30min_MA"], "position"] = 1
df_test.loc[df_test["15min_MA"] < df_test["30min_MA"], "position"] = -1

# Step 6: Backtesting and Performance Evaluation
X_test = df_test.drop(columns=["1. open", "2. high", "3. low", "5. volume", "position"])
y_test = df_test["position"]

# Perform predictions using the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Backtesting Accuracy: {accuracy}")


# Step 7: Execution and Risk Management
def execute_strategy(predictions, capital=100000, transaction_costs=0.001):
    positions = []
    position_size = capital / len(predictions)  # Equal position sizing

    for prediction in predictions:
        if prediction == 1:
            positions.append(position_size)
            capital -= position_size * (1 + transaction_costs)
        elif prediction == -1:
            positions.append(-position_size)
            capital += position_size * (1 - transaction_costs)
        else:
            positions.append(0)

    return positions, capital


positions, final_capital = execute_strategy(predictions)

# Step 8: Calculate Portfolio Value
portfolio_value = final_capital + sum(positions)
print(f"Final Portfolio Value: {portfolio_value}")
