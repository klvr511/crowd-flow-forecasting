import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-03-31 23:00:00", freq="h")
n = len(dates)

hour_effect = np.sin(2 * np.pi * dates.hour / 24) * 500 + 600
weekend_effect = np.where(dates.weekday >= 4, 300, 0)
noise = np.random.normal(0, 100, n)
crowd_count = np.maximum(50, hour_effect + weekend_effect + noise).astype(int)

df = pd.DataFrame({"timestamp": dates, "crowd_count": crowd_count})
df.set_index("timestamp", inplace=True)

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 4).astype(int)

df["lag_1h"] = df["crowd_count"].shift(1)
df["lag_24h"] = df["crowd_count"].shift(24)
df["rolling_mean_6h"] = df["crowd_count"].shift(1).rolling(window=6).mean()

df.dropna(inplace=True)

bottleneck_threshold = df["crowd_count"].quantile(0.90)
df["is_bottleneck"] = (df["crowd_count"] >= bottleneck_threshold).astype(int)

features = ["hour", "day_of_week", "is_weekend", "lag_1h", "lag_24h", "rolling_mean_6h"]
target = "crowd_count"

train_size = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:train_size], df[features].iloc[train_size:]
y_train, y_test = df[target].iloc[:train_size], df[target].iloc[train_size:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = root_mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Model Performance:")
print(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"- Mean Absolute Error (MAE): {mae:.2f}")
print(f"- Bottleneck Threshold: > {bottleneck_threshold:.0f} visitors/hour")

plt.figure(figsize=(12, 5))
plt.plot(y_test.index[:168], y_test.iloc[:168], label="Actual Crowd", color="blue")
plt.plot(y_test.index[:168], predictions[:168], label="Predicted Crowd", color="red", linestyle="--")
plt.axhline(y=bottleneck_threshold, color="orange", linestyle=":", label="Bottleneck Alert Level")
plt.title("Crowd Flow 7-Day Forecast & Bottleneck Detection")
plt.xlabel("Date & Time")
plt.ylabel("Number of Visitors")
plt.legend()
plt.tight_layout()
plt.savefig("crowd_forecast_chart.png")
plt.show()

output_df = X_test.copy()
output_df["actual_crowd"] = y_test
output_df["predicted_crowd"] = predictions.round()
output_df["bottleneck_alert"] = np.where(output_df["predicted_crowd"] >= bottleneck_threshold, "High Alert", "Normal")
output_df.to_csv("crowd_forecast_results.csv")
