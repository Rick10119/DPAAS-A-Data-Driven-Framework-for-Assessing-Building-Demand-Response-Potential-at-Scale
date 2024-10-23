# 利用下载好的数据构建数据集（分不同的州下载）
# %% 1. 导入库
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

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
    # 将其中的 applicability, in.building_subtype 列删去
    upgrade_selected_data_df = upgrade_selected_data_df.drop(columns=['applicability', 'in.building_subtype'])
    
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


    in_sqft = upgrade_selected_data_df['in.sqft']
    in_state = upgrade_selected_data_df['in.state']
    in_state_name = upgrade_selected_data_df['in.state_name']
    calc_weighted_sqft = upgrade_selected_data_df['calc.weighted.sqft']

    np.save(f'downloaded_data/in_sqft_{upgrade_id}.npy', in_sqft)
    np.save(f'downloaded_data/in_state_{upgrade_id}.npy', in_state)
    np.save(f'downloaded_data/in_state_name_{upgrade_id}.npy', in_state_name)
    np.save(f'downloaded_data/calc_weighted_sqft_{upgrade_id}.npy', calc_weighted_sqft)

    # 删除 in.sqft、in.state in.state_name 和 calc.weighted.sqft 项
    upgrade_selected_data_df = upgrade_selected_data_df.drop(columns=['in.sqft', 'in.state', 'in.state_name', 'calc.weighted.sqft'])

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

    # %% 5. 保存数据
    # 保存数据
    np.save(f'downloaded_data/X_{upgrade_id}.npy', X)
    np.save(f'downloaded_data/y_{upgrade_id}.npy', y)