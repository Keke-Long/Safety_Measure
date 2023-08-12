'''
预测分为两个阶段， 首先判断该时间段是否有安全问题（存在指标小于安全阈值的情况），确定存在后再预测该时间段的指标的min， mean， var
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, LSTM, MultiHeadAttention, Flatten, Conv1D, GRU, Dropout
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adamax
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

# Extract feature and target variables
features = ['Speed2', 'Y2']
X = df[features].values
y = df['PET'].values

# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define sequence length for training and testing
seq_length = 30

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


from tensorflow.keras.layers import Dense, Dropout, Multiply
from tensorflow.keras.models import Model

# 定义模型
def multi_task_model():
    inputs = Input(shape=(seq_length, X_train.shape[2]))
    # x = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)  # 添加一个Conv1D层
    x = LSTM(64, return_sequences=True)(x)  # 添加一个LSTM层
    x = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = Flatten()(x)
    # 共享层
    x_shared = Dense(64, activation='relu')(x)
    x_shared = Dropout(0.5)(x_shared)
    x_shared = Dense(64, activation='relu')(x_shared)
    # 分类输出
    out_class = Dense(1, activation='sigmoid', name='classification')(x_shared)
    # 回归输出
    x_reg = Dense(64, activation='relu')(x_shared)
    x_reg = Dropout(0.5)(x_reg)
    x_reg = Dense(64, activation='relu')(x_reg)
    out_reg_raw = Dense(3, activation='linear')(x_reg)
    # 添加一个分支，当预测结果为安全时，该分支不会被激活
    out_reg = Multiply(name='regression')([out_reg_raw, 1 - out_class])
    # 创建模型
    model = Model(inputs=inputs, outputs=[out_class, out_reg])
    # model.compile(optimizer='adam', loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
    #               metrics={'classification': 'accuracy'}) bad result
    model.compile(optimizer=RMSprop(learning_rate=0.005),
                  loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
                  metrics={'classification': 'accuracy'})
    return model


# 准备数据
y_train_binary = np.min(y_train, axis=1) < 5
y_test_binary = np.min(y_test, axis=1) < 5

# 创建并训练模型
model = multi_task_model()
history = model.fit(X_train, {'classification': y_train_binary, 'regression': y_train},
                    validation_data=(X_test, {'classification': y_test_binary, 'regression': y_test}),
                    epochs=50, batch_size=64, verbose=1)

# 预测
predictions = model.predict(X_test)
predictions_binary = predictions[0].flatten() > 0.5
predictions_reg = predictions[1]

# 计算回归预测的MSE
mse_mean = mean_squared_error(y_test[:, 0], predictions_reg[:, 0])
mse_var = mean_squared_error(y_test[:, 1], predictions_reg[:, 1])
mse_min = mean_squared_error(y_test[:, 2], predictions_reg[:, 2])

# 计算二元预测的准确率
accuracy = np.mean(y_test_binary == predictions_binary)

# 打印结果
print(f"Accuracy for binary predictions: {accuracy}")
print(f"MSE for mean predictions: {mse_mean}")
print(f"Average PET value: {np.mean(y_test[:, 0])}")
print(f"MSE for variance predictions: {mse_var}")
print(f"MSE for min predictions: {mse_min}")
