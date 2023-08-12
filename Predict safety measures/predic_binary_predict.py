import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# 加载新数据并预处理
df = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane3.csv")
df.dropna(inplace=True)

# 提取特征和目标变量
features = ['Y2', 'Speed2']
X = df[features].values

# 特征归一化（使用之前保存的scaler进行归一化）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 定义序列长度用于预测
seq_length = 10

# 准备测试数据序列
X_test = []

for _, group_df in df.groupby(['id', 'lane_num']):
    group_len = len(group_df)
    if group_len >= seq_length:
        group_df.reset_index(drop=True, inplace=True)
        X_group_scaled = scaler.fit_transform(group_df[features])
        timestamps = group_df['t'].values

        for i in range(0, group_len - 2 * seq_length, seq_length):
            time_diffs = np.diff(timestamps[i: i + seq_length])
            if np.all(time_diffs < 1):
                X_test.append(X_group_scaled[i: i + seq_length])

X_test = np.array(X_test)

# 加载之前保存的模型
loaded_model = load_model('predict_binary_model.h5')
print("Model loaded successfully.")

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_test)
predictions_binary = predictions.flatten() > 0.5

# 统计每辆车预测出出现安全时间段的占比
num_cars = len(df['id'].unique())
num_safe_cars = np.sum(predictions_binary)
safety_ratio = num_safe_cars / num_cars

# 打印结果
print(f"Total number of cars: {num_cars}")
print(f"Number of cars with safe time periods: {num_safe_cars}")
print(f"Safety ratio: {safety_ratio:.2f}")