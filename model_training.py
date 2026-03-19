import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib

data = pd.read_csv("dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nShape of dataset:")
print(data.shape)

print("\nData Info:")
print(data.info())


print("\nMissing Values:")
print(data.isnull().sum())


le = LabelEncoder()

data['Day_of_Week'] = le.fit_transform(data['Day_of_Week'])
data['Weather'] = le.fit_transform(data['Weather'])

data["Weekend"] = data["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)

print("\nDataset after feature engineering:")
print(data.head())

# 1. Scatter Plot
plt.scatter(data['Expected_Customers'], data['Meals_Consumed'])
plt.xlabel("Expected Customers")
plt.ylabel("Meals Consumed")
plt.title("Customers vs Meals")
plt.show()

# 2. Festival Impact
sns.barplot(x='Festival', y='Meals_Consumed', data=data)
plt.title("Festival vs Meals")
plt.show()

# 3. Heatmap
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

X = data.drop("Meals_Consumed", axis=1)
y = data["Meals_Consumed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

joblib.dump(model, "model.pkl")

print("\nModel trained and saved successfully!")