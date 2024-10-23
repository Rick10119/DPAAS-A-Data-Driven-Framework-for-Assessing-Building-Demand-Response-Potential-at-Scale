# 利用下载好的数据构建数据集
# %% 1. 导入库
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# %% 2. 设置数据集路径并读取数据
# 数据集路径 downloaded_data
data_dir = 'downloaded_data'

# 定义 upgrade_id 列表
upgrade_ids = [19, 20]

for upgrade_id in upgrade_ids:
    # 读取数据 upgrade{upgrade_id}_selected_data.parquet
    upgrade_selected_data_path = os.path.join(data_dir, 'upgrade{}_selected_data.parquet'.format(upgrade_id))
    upgrade_selected_data_df = pd.read_parquet(upgrade_selected_data_path)

    # %% 3. 数据处理（去除一些列，将str转化为连续变量）
    # 将其中的 applicability, in.building_subtype in.state 列删去
    upgrade_selected_data_df = upgrade_selected_data_df.drop(columns=['applicability', 'in.building_subtype', 'in.state'])
    
    # in.wall_construction_type 项，将其转换 feature_transformation.xlsx 中对应的值
    feature_transformation_df = pd.read_excel('feature_transformation.xlsx')

    # 将 in.wall_construction_type 项转换为 feature_transformation.xlsx 中对应的值
    wall_construction_mapping = dict(zip(feature_transformation_df['Original Value'], feature_transformation_df['Transformed Value']))
    upgrade_selected_data_df['in.wall_construction_type'] = upgrade_selected_data_df['in.wall_construction_type'].map(wall_construction_mapping)

    # 将 in.window_type 项转换为 feature_transformation.xlsx 中对应的值
    window_type_mapping = dict(zip(feature_transformation_df['Original Value'], feature_transformation_df['Transformed Value']))
    upgrade_selected_data_df['in.window_type'] = upgrade_selected_data_df['in.window_type'].map(window_type_mapping)

    # 将 in.window_to_wall_ratio_category 项转换为 feature_transformation.xlsx 中对应的值
    window_to_wall_ratio_category_mapping = dict(zip(feature_transformation_df['Original Value'], feature_transformation_df['Transformed Value']))
    upgrade_selected_data_df['in.window_to_wall_ratio_category'] = upgrade_selected_data_df['in.window_to_wall_ratio_category'].map(window_to_wall_ratio_category_mapping)

    # %% 4. 提取特征和标签
    # 从 upgrade_selected_data_df 中划分特征和标签
    # 从当前文件夹中的 selected_output.xlsx 读取标签名称
    selected_output_df = pd.read_excel('selected_output.xlsx')

    # 提取特征和标签
    X = upgrade_selected_data_df.drop(columns=selected_output_df['Output Name'])
    y = upgrade_selected_data_df[selected_output_df['Output Name']]

    # 只选择部分输出 out.qoi.median_daily_peak_jul..kw
    y = y['out.qoi.median_daily_peak_jul..kw']

    # 将数据转换为 numpy 数组，全部是 double 类型
    X = X.values.astype('double')
    y = y.values.astype('double')

    # 数据清洗
    # 缺失值取平均值
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # 删除 y 中大于 0 的值
    X = X[y < 0]
    y = y[y < 0]

    # 删除 y 中小于 5% 分位数的值，以排除异常值
    X = X[y > np.percentile(y, 5)]
    y = y[y > np.percentile(y, 5)]

    # %% 5. 划分训练集、交叉验证集和测试集
    # 划分训练集、交叉验证集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # %% 7. 保存数据
    # 保存数据
    np.save(f'downloaded_data/X_train_{upgrade_id}.npy', X_train)
    np.save(f'downloaded_data/X_val_{upgrade_id}.npy', X_val)
    np.save(f'downloaded_data/X_test_{upgrade_id}.npy', X_test)
    np.save(f'downloaded_data/y_train_{upgrade_id}.npy', y_train)
    np.save(f'downloaded_data/y_val_{upgrade_id}.npy', y_val)
    np.save(f'downloaded_data/y_test_{upgrade_id}.npy', y_test)