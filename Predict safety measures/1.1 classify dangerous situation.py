'''
input: individual veh trajectory
output: whether PET < safety threshold
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, LSTM, MultiHeadAttention, Flatten, Conv1D, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import RMSprop, schedules
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random


def get_random_subset(X, y, desired_size):
    if len(X) < desired_size:
        raise ValueError("Desired size is larger than the dataset size.")
    indices = random.sample(range(len(X)), desired_size)
    return X[indices], y[indices]


# Key parameters
Safe_threshold = 2
seq_length = 15 #序列长度
#Sample_num = 10300

# 加载数据并预处理
lane_numbers = [1, 2, 3]
df = pd.DataFrame()
for lane_number in lane_numbers:
    file_path = f'../Data/HighSIM/HighSim_safety_measures_lane{lane_number}.csv'
    df_onelane = pd.read_csv(file_path)
    df = pd.concat([df, df_onelane], ignore_index=True)

# 设置数据精度 为1Hz
df = df[df['t'].apply(lambda x: x == int(x))].reset_index(drop=True)
# 设置数据精度 为5Hz
# df_5Hz = df.iloc[::6].copy()
# df_5Hz.reset_index(drop=True, inplace=True)
# df = df_5Hz

# 提取特征和目标变量
features = ['Speed2', 'A2']
X = df[features].values

# 特征归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 准备训练和测试数据序列
X_data, y_data = [], []

for _, group_df in df.groupby(['id', 'lane_num']):
    group_len = len(group_df)
    if group_len >= seq_length:
        group_df.reset_index(drop=True, inplace=True)
        X_group_scaled = scaler.fit_transform(group_df[features])
        y_group = np.clip(group_df['PET'].values, a_min=0, a_max=Safe_threshold)
        timestamps = group_df['t'].values

        for i in range(0, group_len - seq_length, seq_length):
            time_diffs = np.diff(timestamps[i: i + seq_length])
            if np.all(time_diffs < 1.1):
                X_data.append(X_group_scaled[i: i + seq_length])
                y_data.append(np.min(y_group[i: i + seq_length]) < Safe_threshold)

X_data = np.array(X_data)
print("X_data.shape", X_data.shape)
y_data = np.array(y_data)

# Set total data amount
# X_data, y_data = get_random_subset(X_data, y_data, Sample_num)
# print("X_data.shape", X_data.shape)

# 先划分临时数据和测试集
X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data, test_size=0.15, random_state=42)
# 再从临时数据中划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# 定义模型
def binary_classification_model():
    inputs = Input(shape=(seq_length, X_train.shape[2]))
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x) # Dropout层
    x = Dense(64, activation='relu')(x)
    out_class = Dense(1, activation='sigmoid', name='classification')(x)
    model = Model(inputs=inputs, outputs=out_class)
    # 使用学习率衰减策略
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.003,
                                             decay_steps=500,
                                             decay_rate=0.9)
    optimizer = RMSprop(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    return model


def binary_classification_model2():
    inputs = Input(shape=(seq_length, X_train.shape[2]))

    # Adding more convolution layers
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)

    x = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    out_class = Dense(1, activation='sigmoid', name='classification')(x)

    model = Model(inputs=inputs, outputs=out_class)

    # Adjusting learning rate and decay steps
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95)
    optimizer = RMSprop(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    return model

def binary_classification_model3():
    inputs = Input(shape=(seq_length, X_train.shape[2]))

    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)

    x = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)
    x = Flatten()(x)

    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)
    out_class = Dense(1, activation='sigmoid', name='classification')(x)

    model = Model(inputs=inputs, outputs=out_class)
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=1000, decay_rate=0.9)
    optimizer = RMSprop(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    return model

# 创建并训练模型
model = binary_classification_model3()

# 定义早期停止回调
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, verbose=2)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=64, verbose=2, callbacks=[early_stopping])

# 保存模型
model.save('./models/1.1 classify dangerous situation.h5')
print("Model saved successfully")


# 预测
predictions = model.predict(X_test)
predictions_binary = (predictions.flatten() > 0.5).astype(int)
y_test_int = y_test.astype(int)

# 计算TP, TN, FP, FN, 正确率
TP = np.sum((predictions_binary == 1) & (y_test_int == 1))
TN = np.sum((predictions_binary == 0) & (y_test_int == 0))
FP = np.sum((predictions_binary == 1) & (y_test_int == 0))
FN = np.sum((predictions_binary == 0) & (y_test_int == 1))
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
specificity = TN / (TN + FP)
f1_score = 2 * (precision * recall) / (precision + recall)

# 打印各种情况的数量及总正确率
with open(f"./results/predict_results.txt", 'a') as f:
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    f.write(f'{current_time}\n')
    f.write(f"Safe_threshold = {Safe_threshold}\n\n")
    f.write(f"True Positives (TP): {TP}\n")
    f.write(f"True Negatives (TN): {TN}\n")
    f.write(f"False Positives (FP): {FP}\n")
    f.write(f"False Negatives (FN): {FN}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"F1 Score: {f1_score:.4f}\n\n\n")
