import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
# Load and preprocess data
file_path = os.path.join(os.path.dirname(__file__), './data/ev_charging_data.csv')
data = pd.read_csv(file_path)

X = data[['remaining_battery', 'drain_rate', 'remaining_range', 'estimated_time_left', 'time_to_station', 'distance_to_station']]
y = data['priority']

# Add noise to the data
noise_factor = 0.5
X_noisy = X + noise_factor * np.random.normal(size=X.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

# RandomForestRegressor with initial parameters
model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
# Example of k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf)
print(f"Cross-validated R^2 scores: {scores}")


selector = SelectFromModel(estimator=model, threshold='median')
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
# Optimized RandomForestRegressor with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}
print("grid searching")
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_
print("grid searching over")
# Evaluate best model performance
y_pred = best_model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the best model
model_path = 'C:/Users/Diya/projects/ev_charging_model_rf.joblib'
joblib.dump(best_model, model_path)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Priority')
plt.ylabel('Predicted Priority')
plt.title('Actual vs. Predicted Priority')
plt.show()

#testing
data_points = {
    'car_id': [101, 102],
    'remaining_battery': [10,10],
    'drain_rate': [5, 8],
    'remaining_range': [18,10],
    'estimated_time_left': [10,6],
    'time_to_station': [10,10],
    'distance_to_station': [10,10]
}

input_df = pd.DataFrame(data_points)
input_df.fillna(input_df.mean(), inplace=True)
X_input = input_df[['remaining_battery', 'drain_rate', 'remaining_range', 'estimated_time_left', 'time_to_station', 'distance_to_station']]

predicted_priorities = best_model.predict(X_input)

input_df['predicted_priority'] = predicted_priorities

priority_list = input_df.sort_values(by='predicted_priority', ascending=False)
print(priority_list)