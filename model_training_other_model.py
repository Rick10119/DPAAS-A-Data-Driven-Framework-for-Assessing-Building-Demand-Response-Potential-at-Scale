# Description: This script trains and evaluates other models (Random Forest, Gradient Boosting, Support Vector Regressor, and Neural Network) on the data of two upgrades (Load Shedding and Load Shift). The nRMSE results are visualized using box plots and tables.

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# %% 定义 upgrade_id 列表
upgrade_ids = [19, 20]

# 存储所有 nRMSE 结果
all_nrmse_results = {}

for upgrade_id in upgrade_ids:
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 特征选择 - SelectKBest
    k = 6  # 选择前 k 个特征
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # 定义模型及其最优参数
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf', C=10),
        'Neural Network': Sequential([
            Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
    }

    # 训练和评估模型
    nrmse_results = {}
    for name, model in models.items():
        if name == 'Neural Network':
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_selected, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_val_selected, y_val))
            y_test_pred = model.predict(X_test_selected).flatten()
        else:
            model.fit(X_train_selected, y_train)
            y_test_pred = model.predict(X_test_selected)

        # 计算每个数据点的误差
        errors = np.abs(y_test - y_test_pred) / (np.max(y_test) - np.min(y_test))
        nrmse_results[name] = errors

    # 存储当前 upgrade_id 的 nRMSE 结果
    all_nrmse_results[upgrade_id] = nrmse_results

# %% 绘制箱线图
selected_upgrade_id = 19  # 你可以在这里选择不同的 upgrade_id

# 创建子图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True)

# 增大所有字号
plt.rcParams.update({'font.size': 20})

# 增大坐标轴刻度的字体
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=20)

# 模型名字缩写
model_abbr = {
    'Linear Regression': 'LR',
    'Random Forest': 'RF',
    'Gradient Boosting': 'GB',
    'Support Vector Regressor': 'SVR',
    'Neural Network': 'NN'
}

# 绘制每个 upgrade_id 的箱线图
for ax, upgrade_id in zip(axes, all_nrmse_results.keys()):
    labels = []
    data = []
    for name in all_nrmse_results[upgrade_id].keys():
        labels.append(model_abbr.get(name, name))  # 使用缩写
        data.append(all_nrmse_results[upgrade_id][name])
    
    # 绘制箱线图，显示离群值
    box = ax.boxplot(data, patch_artist=False, labels=labels, flierprops=dict(markerfacecolor='white'))
    
    # 标出中位数
    medians = [np.median(d) for d in data]
    for i, median in enumerate(medians):
        ax.text(i + 1, median, f'{median:.2f}', ha='center', va='bottom', color='blue')

    if upgrade_id == 19:
        ax.set_title('Load Shedding')
    if upgrade_id == 20:
        ax.set_title('Load Shift')    
    ax.set_ylim(0, 0.6)  # 设置纵轴范围

    # 设置纵轴为百分数
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# %%
selected_upgrade_id = 19  # 你可以在这里选择不同的 upgrade_id

# 绘制表格
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table_data = [[model_abbr.get(name, name), f'{np.mean(all_nrmse_results[selected_upgrade_id][name])*100:.3f}%', f'{np.median(all_nrmse_results[selected_upgrade_id][name])*100:.3f}%'] for name in all_nrmse_results[selected_upgrade_id].keys()]
table = ax.table(cellText=table_data, colLabels=['Model', 'Mean nRMSE', 'Median nRMSE'], cellLoc='center', loc='center')

plt.show()
# %%
