import pandas as pd
import gc  # 引入垃圾回收模块

#
df1 = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane1.csv")
df2 = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane2.csv")
df3 = pd.read_csv("../Data/HighSIM/HighSim_safety_measures_lane3.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)
columns_to_keep = ["id", "t", "Y2", "Speed2", "A2", "lane_num", "PET"]
df_selected_columns = df[columns_to_keep]
# print(max(df_selected_columns['lane_num']))
df_selected_columns.to_csv("../Data/HighSIM/HighSim_safety_measures_lane.csv", index=False)

# 删除原始的 DataFrame，并强制进行垃圾回收
del df1, df2, df3
gc.collect()

#
# df1 = pd.read_csv("../Data/HighSIM/risk_values_lane1.csv")
# df2 = pd.read_csv("../Data/HighSIM/risk_values_lane2.csv")
# df3 = pd.read_csv("../Data/HighSIM/risk_values_lane3.csv")
# df1['lane_num'] = 1
# df2['lane_num'] = 2
# df3['lane_num'] = 3
# df = pd.concat([df1, df2, df3], ignore_index=True)
# df.to_csv("../Data/HighSIM/risk_values_lane.csv", index=False)
#
# # 删除原始的 DataFrame，并强制进行垃圾回收
# del df1, df2, df3
# gc.collect()