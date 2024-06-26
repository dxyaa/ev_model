import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import os

#file_path = 'C:/Users/Diya/projects/ev_charging_data.csv'
file_path= os.path.join(os.path.dirname(__file__), './data/ev_charging_data.csv')
data = pd.read_csv(file_path)


X = data[['remaining_battery', 'drain_rate', 'remaining_range', 'estimated_time_left', 'time_to_station', 'distance_to_station']]
y = data['priority']


noise_factor = 0.5
X_noisy = X + noise_factor * np.random.normal(size=X.shape)


X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)


model.fit(X_train, y_train)

model_path = 'C:/Users/Diya/projects/ev_charging_model_rf.joblib'
joblib.dump(model, model_path)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Priority')
plt.ylabel('Predicted Priority')
plt.title('Actual vs. Predicted Priority')
plt.show()

#testing
data_points = {
    'car_id': [101, 102, 103, 104, 105],
    'remaining_battery': [50, 20, 80, 30, 60],
    'drain_rate': [5, 8, 3, 6, 4],
    'remaining_range': [200, 100, 300, 150, 250],
    'estimated_time_left': [100, 50, 200, 75, 125],
    'time_to_station': [30, 15, 45, 20, 25],
    'distance_to_station': [15, 5, 20, 10, 12]
}

input_df = pd.DataFrame(data_points)
input_df.fillna(input_df.mean(), inplace=True)
X_input = input_df[['remaining_battery', 'drain_rate', 'remaining_range', 'estimated_time_left', 'time_to_station', 'distance_to_station']]

predicted_priorities = model.predict(X_input)

input_df['predicted_priority'] = predicted_priorities

priority_list = input_df.sort_values(by='predicted_priority', ascending=False)
print(priority_list)