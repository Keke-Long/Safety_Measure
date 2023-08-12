import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
trajectory_data = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane.csv")
risk_values = pd.read_csv("../Data/HighSIM/risk_values_lane.csv")
n = risk_values['Risk'].count()
print(f'一共有{n}辆车有risk值')

# Merge the two dataframes on the car id and lane_num
merged_data = pd.merge(trajectory_data, risk_values, on='id')

# Drop any rows with missing Risk values
merged_data.dropna(subset=['Risk'], inplace=True)

# Sort the data by id and time
merged_data.sort_values(['id', 't'], inplace=True)

# Group the data by id
grouped = merged_data.groupby('id')
n = len(grouped)
print(f'一共有{n}个grouped样本')

# Check how many cars have at least 47 data points after merging and cleaning the data
valid_ids = merged_data.groupby('id').filter(lambda x: len(x) >= 10)['id'].unique()

# Compute the sequence lengths for each car
sequence_lengths = merged_data.groupby('id').size()

# Find the maximum sequence length among cars with 'Risk' value
max_sequence_length = sequence_lengths[valid_ids].max()

# Initialize lists to hold the input sequences and the target values
X = []
y = []

# Create the input sequences and the target values
for car_id in valid_ids:
    group = grouped.get_group(car_id)

    # Pad the sequence to the maximum sequence length
    pad_size = max_sequence_length - len(group)
    padded_sequence = np.pad(group[['Y2', 'PET']].values, ((0, pad_size), (0, 0)), 'constant',
                             constant_values=0)

    # Add the sequence to the list of input sequences
    X.append(padded_sequence)

    # Add the risk value to the list of target values
    y.append(group['Risk'].values[0])

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Create a StandardScaler object
scaler = StandardScaler()
# Reshape the input data to 2D because StandardScaler expects 2D data
X_reshaped = X.reshape(-1, X.shape[-1])
# Fit the scaler to the training data and transform the data
X_scaled = scaler.fit_transform(X_reshaped)
# Reshape the data back to 3D
X = X_scaled.reshape(X.shape)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Define the LSTM model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(max_sequence_length, 3), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(40, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Fit the model to the training data
early_stopping = EarlyStopping(monitor='loss', min_delta=1, patience=3)
history = model.fit(X_train, y_train, epochs=10, verbose=1, callbacks=[early_stopping])

# Use the model to predict the risk values for the test set
y_pred = model.predict(X_test)
print(y_pred)

# Compute the root mean squared error of the predictions
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('rmse=', rmse)

print("Number of training samples:", X_train.shape[0])
print("Number of test samples:", X_test.shape[0])


plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()
