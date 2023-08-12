import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the trajectory data
trajectory_data = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane.csv")

# Load the risk values
risk_values = pd.read_csv("../Data/HighSIM/risk_values_lane.csv")

# Merge the two dataframes on the car id
merged_data = pd.merge(trajectory_data, risk_values, on=['id', 'lane_num'])

# Drop any rows with missing values
merged_data.dropna(inplace=True)

# Separate the input features and the target variable
X = merged_data[['Y2', 'Speed2', 'A2']]
y = merged_data['Risk']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Use the model to predict the risk values for the test set
y_pred = rf.predict(X_test)

# Compute the root mean squared error of the predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('rmse=',rmse)

