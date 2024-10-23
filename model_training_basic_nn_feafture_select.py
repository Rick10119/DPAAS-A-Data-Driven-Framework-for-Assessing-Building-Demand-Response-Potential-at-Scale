# 利用下载好的数据训练一个简单的神经网络
# %% 1. 导入库
import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error

# %% 定义 upgrade_id
upgrade_id = 19

# 读取存储的数据
data_dir = 'downloaded_data'
X_train = np.load(os.path.join(data_dir, f'X_train_{upgrade_id}.npy'))
X_val = np.load(os.path.join(data_dir, f'X_val_{upgrade_id}.npy'))
X_test = np.load(os.path.join(data_dir, f'X_test_{upgrade_id}.npy'))
y_train = np.load(os.path.join(data_dir, f'y_train_{upgrade_id}.npy'))
y_val = np.load(os.path.join(data_dir, f'y_val_{upgrade_id}.npy'))
y_test = np.load(os.path.join(data_dir, f'y_test_{upgrade_id}.npy'))

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 特征选择
n_components = 6
selector = SelectKBest(score_func=f_regression, k=n_components)
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

# 获取被选中的特征索引
selected_features = selector.get_support(indices=True)
print(f'Selected feature indices: {selected_features}')

# 打印各个特征的 F 值和 p 值
feature_scores = selector.scores_
feature_pvalues = selector.pvalues_
for i, (score, pvalue) in enumerate(zip(feature_scores, feature_pvalues)):
    print(f'Feature {i}: F-value = {score:.3f}, p-value = {pvalue:.3f}')

# %% 创建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 训练神经网络
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=100
)

# 评估模型
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'Train NRMSE: {train_rmse:.2f}')
print(f'Validation NRMSE: {val_rmse:.2f}')
print(f'Test NRMSE: {test_rmse:.2f}')

# %% 画出测试集中预测值和真实值的对比（前100个点，从大到小排列）
temp = 50
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 15})
plt.plot(-y_test[temp + 1 : temp + 95], label='True')
plt.plot(-y_test_pred[temp + 1 : temp + 95], label='Predicted')
plt.ylabel('Potential Load Shedding (kW)')
plt.xlabel('Building ID in Test Set')
plt.legend()

plt.show()

# %% 画出预测值的每个点对应的排序

# 确保 y_test 和 y_test_pred 是1维数组
y_test = y_test.flatten()
y_test_pred = y_test_pred.flatten()
# 获取排序后的索引
sorted_indices = np.argsort(y_test)

# 创建一个与 y_test 维度相同的数组，用于存储排名
y_test_order = np.empty_like(sorted_indices)

# 根据排序后的索引生成排名
y_test_order[sorted_indices] = np.arange(len(y_test))

# 将排名从 0-based 转换为 1-based
y_test_order = y_test_order + 1

# 对y_test_pred进行相同操作
sorted_indices = np.argsort(y_test_pred)
y_test_pred_order = np.empty_like(sorted_indices)
y_test_pred_order[sorted_indices] = np.arange(len(y_test_pred))
y_test_pred_order = y_test_pred_order + 1

# %% 画出预测值的每个点对应的排序，计算相关系数并画在图上
plt.figure(figsize=(10, 5))
plt.scatter(y_test_order, y_test_pred_order)
plt.xlabel('True Order')
plt.ylabel('Predicted Order')
plt.plot([0, len(y_test)], [0, len(y_test)], color='red')
plt.title(f'Correlation: {np.corrcoef(y_test_order, y_test_pred_order)[0, 1]:.2f}')
plt.rcParams.update({'font.size': 15})
plt.show()

# %% 在一幅图里按照排序后的预测值和真值累加的值
plt.figure(figsize=(10, 5))
plt.plot(-np.cumsum(y_test[sorted_indices]), label='True')
plt.plot(-np.cumsum(y_test_pred[sorted_indices]), label='Predicted')
plt.xlabel('Number of Selected Buildings')
plt.ylabel('Cumulative Potential-Load Shed (kW)')
plt.legend()
plt.rcParams.update({'font.size': 15})
plt.show()


# %%
