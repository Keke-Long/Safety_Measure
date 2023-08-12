'''
model predicting the mean,var,min of PET
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, LSTM, MultiHeadAttention, Flatten, Conv1D, GRU
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


lane_numbers = [1, 2]
df = pd.DataFrame()
for lane_number in lane_numbers:
    file_path = f'../Data/HighSIM/HighSim_safety_measures_lane{lane_number}.csv'
    df_onelane = pd.read_csv(file_path)
    df = pd.concat([df, df_onelane], ignore_index=True)

df.dropna(inplace=True)

# limiting the precision of the data
for col in df.columns:
    if df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float16)


# Extract feature and target variables
features = ['Speed1', 'Y1', 'Speed2', 'Y2', 'Speed4', 'Y4']
X = df[features].values
y = df['foll_PET'].values

# check PET value
y = np.clip(df['foll_PET'].values, a_min=0, a_max=200)
# plt.hist(y, bins=500, color='blue')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

y = np.clip(df['foll_PET'].values, a_min=0, a_max=5)

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define sequence length for training and testing
seq_length = 500

# Prepare train and test data sequences
X_train, X_test, y_train, y_test = [], [], [], []

for _, group_df in df.groupby(['id', 'lane_num']):
    group_len = len(group_df)
    if group_len >= seq_length:
        group_df.reset_index(drop=True, inplace=True)
        X_group_scaled = scaler.fit_transform(group_df[features])
        #y_group = group_df['foll_PET'].values
        y_group = np.clip(group_df['foll_PET'].values, a_min=0, a_max=5)
        timestamps = group_df['t'].values

        for i in range(0, group_len - 2*seq_length, seq_length):
            time_diffs = np.diff(timestamps[i: i + seq_length])
            if np.all(time_diffs < 1):
                variance = np.var(y_group[i: i + seq_length])
                if np.isnan(variance):
                    continue  # Skip this sequence if the variance is NaN

                X_train.append(X_group_scaled[i: i + seq_length])
                y_train.append([np.mean(y_group[i: i + seq_length]), variance, np.min(y_group[i: i + seq_length])])

        variance_test = np.var(y_group[-seq_length:])
        if np.isnan(variance_test):
            continue  # Skip this sequence if the variance is NaN
        X_test.append(X_group_scaled[-seq_length:])
        y_test.append([np.mean(y_group[-seq_length:]), variance_test, np.min(y_group[-seq_length:])])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# Plot histograms
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].hist(y_train[:, 0], bins=50, color='blue')
axs[0].set_title('Mean of y_group')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

axs[1].hist(y_train[:, 1], bins=50, color='green')
axs[1].set_title('Variance of y_group')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

axs[2].hist(y_train[:, 2], bins=50, color='red')
axs[2].set_title('Minimum of y_group')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()




# Define a function to create models
def model1():
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def model2():
    model_var = Sequential()
    model_var.add(Dense(64, input_shape=(seq_length, X_train.shape[2]), activation='relu'))
    model_var.add(Flatten())
    model_var.add(Dense(units=1))
    model_var.compile(optimizer='adam', loss='mse')

def model3():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model

def model_GRU():
    model = Sequential()
    model.add(GRU(units=64, return_sequences=False, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model

def model_Transformer():
    inputs = Input(shape=(seq_length, X_train.shape[2]))
    x = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    x = Flatten()(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and train models for mean, variance and min
# model_mean = model1()
# model_mean.fit(X_train, y_train[:, 0], epochs=5, batch_size=64, verbose=1)
#
# model_var = model1()
# model_var.fit(X_train, y_train[:, 1], epochs=5, batch_size=64, verbose=1)

model_min = model1()
model_min.fit(X_train, y_train[:, 2], epochs=50, batch_size=64, verbose=1)

# Make predictions
# predictions_mean = model_mean.predict(X_test).flatten()
# predictions_var  = model_var.predict(X_test).flatten()
predictions_min  = model_min.predict(X_test).flatten()

# Compute MSE
# mse_mean = mean_squared_error(y_test[:, 0], predictions_mean.flatten())
# mse_var = mean_squared_error(y_test[:, 1], predictions_var.flatten())
rmse_min = np.sqrt(mean_squared_error(y_test[:, 2], predictions_min.flatten()))

# Print results
print(f"Average PET value: { np.mean(y_test[:, 0]) }")
# print(f"MSE for mean predictions: {mse_mean}")
# print(f"MSE for variance predictions: {mse_var}")
print(f"MSE for min predictions: {rmse_min}")