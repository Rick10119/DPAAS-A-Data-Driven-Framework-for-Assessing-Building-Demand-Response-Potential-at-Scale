# 提取同一个气候区的建筑用能数据
# %% 1. Import Libraries
import os.path
import boto3  # This is not called directly, but must be installed for Pandas to read files from S3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% 设置要下载的数据信息
dataset_year = '2024'
dataset_name = 'comstock_amy2018_release_1'
dataset_path = f's3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{dataset_year}/{dataset_name}'
upgrade_id = 19
upgrade_name = 'Demand Flexibility, Thermostat Control, Load Shed'
# upgrade_id = 20
# upgrade_name = 'Demand Flexibility, Thermostat Control, Load Shift'

# %% 2. 下载数据（parquet-baseline）
# 设置数据读取路径(元数据和年度数据-baseline)
baseline_metadata_and_annual_path = f'{dataset_path}/metadata_and_annual_results/national/parquet/baseline_metadata_and_annual_results.parquet'

upgrade_metadata_and_annual_path = f'{dataset_path}/metadata_and_annual_results/national/parquet/upgrade{upgrade_id}_metadata_and_annual_results.parquet'

# 读取数据（parquet-baseline）
baseline_meta_and_annual_df = pd.read_parquet(baseline_metadata_and_annual_path)
# 展示数据（前五行）
baseline_meta_and_annual_df.head()

# 读取数据（parquet-upgrade）
upgrade_meta_and_annual_df = pd.read_parquet(upgrade_metadata_and_annual_path)
# 展示数据（前五行）
upgrade_meta_and_annual_df.head()

# %% 3. 提取数据(读取特征和输出名称)
# 从当前文件夹中的seleted_features.xlsx读取特称名称
selected_features_df = pd.read_excel('selected_features_us.xlsx')

# 从当前文件夹中的seleted_output.xlsx读取输出名称
selected_output_df = pd.read_excel('selected_output.xlsx')

# 从upgrade_meta_and_annual_df中提取selected_features_df中包含的特征的列
upgrade_selected_features_df = upgrade_meta_and_annual_df[selected_features_df['Feature Name']]

# %% 4. 提取数据（筛选数据）
# 从upgrade_selected_features_df中提取applicability==true, in.building_subtype==largeoffice_nodatacenter的数据
applicability_series = upgrade_selected_features_df['applicability']
building_subtype_series = upgrade_selected_features_df['in.building_subtype']

# Align the index of the boolean Series with the DataFrame
applicability_series = applicability_series.reindex(upgrade_meta_and_annual_df.index)
building_subtype_series = building_subtype_series.reindex(upgrade_meta_and_annual_df.index)

# Apply the boolean conditions sequentially
upgrade_selected_features_df = upgrade_selected_features_df[applicability_series == True]
upgrade_selected_features_df = upgrade_selected_features_df[building_subtype_series == 'largeoffice_nodatacenter']

# 对baseline_meta_and_annual_df，只保留upgrade_selected_features_df对应的那些行和列，保存到baseline_selected_features_df
baseline_selected_features_df = baseline_meta_and_annual_df.loc[upgrade_selected_features_df.index]

# 数据处理(差值)
# 用upgrade_selected_features_df中的数据减去baseline_selected_features_df中的数据（selected_output_df对应的列）
upgrade_selected_features_df[selected_output_df['Output Name']] = upgrade_selected_features_df[selected_output_df['Output Name']] - baseline_selected_features_df[selected_output_df['Output Name']]


# %% 存储数据
# 保存数据到downloaded_data文件夹，文件名为upgrade{upgrade_id}_selected_data.parquet
upgrade_selected_features_df.to_parquet(f'downloaded_data/upgrade{upgrade_id}_selected_data.parquet')

# %%
