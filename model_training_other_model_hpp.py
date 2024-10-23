# test the performance of other models with hyperparameter optimization

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import os

# Load data
X_train = np.load('downloaded_data/X_train.npy', allow_pickle=True)
X_val = np.load('downloaded_data/X_val.npy', allow_pickle=True)
X_test = np.load('downloaded_data/X_test.npy', allow_pickle=True)
y_train = np.load('downloaded_data/y_train.npy', allow_pickle=True)
y_val = np.load('downloaded_data/y_val.npy', allow_pickle=True)
y_test = np.load('downloaded_data/y_test.npy', allow_pickle=True)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Feature selection
k = 8
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Define hyperparameter grids
param_grids = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'Support Vector Regressor': {
        'model': SVR(),
        'params': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10]
        }
    }
}

# Perform hyperparameter optimization and collect nRMSE
results = {}
best_params = {}
for name, config in param_grids.items():
    param_file = f'downloaded_data/{name}_best_params.npy'
    if os.path.exists(param_file):
        best_params[name] = np.load(param_file, allow_pickle=True).item()
    else:
        grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train_selected, y_train)
        best_params[name] = grid_search.best_params_
        np.save(param_file, best_params[name])
    
    best_model = config['model'].set_params(**best_params[name])
    best_model.fit(X_train_selected, y_train)
    
    y_test_pred = best_model.predict(X_test_selected)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    nrmse = test_rmse / np.mean(y_test)
    results[name] = nrmse

# Print the best parameters for each model
for name, params in best_params.items():
    print(f"Best parameters for {name}: {params}")