#%%
import numpy as np
import sklearn.linear_model as skl
import gradescope_utils
import pandas as pd
import scipy

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

class FeatureSelection:
    def rank_correlation(self, x, y):
        ranks = np.zeros(y.shape[0])
        #your code here
        corr_all = []
        for i in range(x.shape[1]):
            corr, _ = pearsonr(x[:, i], y)
            corr_all.append(corr)
        ranks = np.argsort(corr_all)[::-1]    
  
        return ranks

    def lasso(self, x, y):
        ranks = np.zeros(y.shape[0])
        # your code here
        lasso = skl.Lasso(alpha=0.01) 
        lasso.fit(x, y)
        
        coefs = np.abs(lasso.coef_)
        non_zero_ind = np.where(coefs != 0)[0]
        ranks = np.argsort(coefs[non_zero_ind])[::-1]

        return ranks

    def stepwise(self, x, y):
        ranks = np.zeros(y.shape[0])
        # your code here
        n_features = x.shape[1]
        enlisted_features = list(range(n_features))
        
        selected_features = []
        current_rmse = np.inf # starts with inf
        
        while enlisted_features: # update while lookup the features
            best_feature = None
            best_rmse = current_rmse
            
            for feature in enlisted_features: # LR things in each features
                trial_features = selected_features + [feature]
                LR = skl.LinearRegression().fit(x[:, trial_features], y)
                y_pred = LR.predict(x[:, trial_features])
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                if rmse < best_rmse: #update best
                    best_rmse = rmse
                    best_feature = feature
            
            if best_rmse < current_rmse: # remove, append
                selected_features.append(best_feature)
                enlisted_features.remove(best_feature)
                current_rmse = best_rmse
            else:
                break
        ranks = np.array(selected_features)
        return ranks

class Regression:
    def ridge_lr(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        ridge = skl.Ridge(alpha=1.0)
        ridge.fit(train_x, train_y)
        test_prob = ridge.predict(test_x)
        
        rmse = np.sqrt(mean_squared_error(test_y, test_prob))
        r2 = r2_score(test_y, test_prob)
        return test_prob, rmse, r2

    def tree_regression(self, train_x, train_y, test_x, test_y, max_depth, min_items):
        test_prob = np.zeros(test_x.shape[0])
        
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_items)
        tree.fit(train_x, train_y)
        test_prob = tree.predict(test_x)
        
        rmse = np.sqrt(mean_squared_error(test_y, test_prob))
        r2 = r2_score(test_y, test_prob)
        return test_prob, rmse, r2, tree

#%%

#%%
train = pd.read_csv('./energydata/energy_train.csv', parse_dates=['date'], dayfirst=True)
val = pd.read_csv('./energydata/energy_val.csv', parse_dates=['date'], dayfirst=True)
test = pd.read_csv('./energydata/energy_test.csv', parse_dates=['date'], dayfirst=True)

feature_names = train.drop(['Appliances', 'date'], axis=1).columns.tolist()
print(len(feature_names))

# Separate features and target variable for training, validation, and test sets
train_x_pd = train.drop(['Appliances', 'date'], axis=1)
valid_x_pd = val.drop(['Appliances', 'date'], axis=1)
test_x_pd = test.drop(['Appliances', 'date'], axis=1)

train_x, train_y = train_x_pd.to_numpy(), train['Appliances'].values
valid_x, valid_y = valid_x_pd.to_numpy(), val['Appliances'].values
test_x, test_y = test_x_pd.to_numpy(), test['Appliances'].values

print(train_x.shape, valid_x.shape, test_x.shape)
print(train_y.shape, valid_y.shape, test_y.shape)
#%%

#%%
fs = FeatureSelection()
reg = Regression()

# a
ranked_features_corr = fs.rank_correlation(train_x, train_y)
ten_corr = [feature_names[i] for i in ranked_features_corr][:10]
print("Ranked Features by Correlation:", ranked_features_corr)

# b
ranked_features_lasso = fs.lasso(train_x, train_y)
ten_lasso = [feature_names[i] for i in ranked_features_lasso][:10]
print("Ranked Features by Lasso:", ranked_features_lasso)

# c
ranked_features_stepwise = fs.stepwise(train_x, train_y)
ten_stepwise = [feature_names[i] for i in ranked_features_stepwise][:10]
print("Ranked Features by Stepwise Selection:", ranked_features_stepwise)
#%%

#%%
# Create an instance of Regression
reg = Regression()

# applying selected features

# Ridge regression on full feature set
ridge_pred_full, ridge_rmse_full, ridge_r2_full = reg.ridge_lr(train_x, train_y, test_x, test_y)

train_corr = train_x_pd[ten_corr].to_numpy()
test_corr = test_x_pd[ten_corr].to_numpy()
ridge_pred_corr, ridge_rmse_corr, ridge_r2_corr = reg.ridge_lr(train_corr, train_y, test_corr, test_y)

train_lasso = train_x_pd[ten_lasso].to_numpy()
test_lasso = test_x_pd[ten_lasso].to_numpy()
ridge_pred_lasso, ridge_rmse_lasso, ridge_r2_lasso = reg.ridge_lr(train_lasso, train_y, test_lasso, test_y)

train_step = train_x_pd[ten_stepwise].to_numpy()
test_step = test_x_pd[ten_stepwise].to_numpy()
ridge_pred_step, ridge_rmse_step, ridge_r2_step = reg.ridge_lr(train_step, train_y, test_step, test_y)

# Report results in a table format
result_df = pd.DataFrame({
    'Feature Selection Method': ['No Selection', '1a (Correlation)', '1b (Lasso)', '1c (Stepwise)'],
    'RMSE (Test Set)': [ridge_rmse_full, ridge_rmse_corr, ridge_rmse_lasso, ridge_rmse_step],
    'R2 (Test Set)': [ridge_r2_full, ridge_r2_corr, ridge_r2_lasso, ridge_r2_step]
})

# Display results
result_df

# # e
# ridge_pred, ridge_rmse, ridge_r2 = reg.ridge_lr(train_x, train_y, test_x, test_y)
# print(ridge_pred)
# print("Ridge Regression - RMSE:", ridge_rmse, "R2:", ridge_r2)

# # g
# tree_pred, tree_rmse, tree_r2 = reg.tree_regression(train_x, train_y, test_x, test_y, max_depth=3, min_items=5)
# print(tree_pred)
# print("Decision Tree Regression - RMSE:", tree_rmse, "R2:", tree_r2)
#%%

#%%
# Create an instance of Regression
train_corr = train_x_pd[ten_corr].to_numpy()
valid_corr = valid_x_pd[ten_corr].to_numpy()
test_corr = test_x_pd[ten_corr].to_numpy()

train_lasso = train_x_pd[ten_lasso].to_numpy()
valid_lasso = valid_x_pd[ten_lasso].to_numpy()
test_lasso = test_x_pd[ten_lasso].to_numpy()

train_step = train_x_pd[ten_stepwise].to_numpy()
valid_step = valid_x_pd[ten_stepwise].to_numpy()
test_step = test_x_pd[ten_stepwise].to_numpy()

max_depth_values = [3, 5, 7, 10, 15, 20]
min_items_values = [1, 2, 5, 10]

# applying selected features

best_rmse = np.inf
best_r2 = -np.inf
best_config = {'max_depth': None, 'min_items': None}
reg = Regression()
for max_depth in max_depth_values:
    for min_items in min_items_values:
        # Train and validate the decision tree on the validation set
        _, rmse, r2,_  = reg.tree_regression(train_x, train_y, valid_x, valid_y, max_depth=max_depth, min_items=min_items)
        
        # If current model is better, update the best configuration
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_config['max_depth'] = max_depth
            best_config['min_items'] = min_items
print("Best Configuration:")
print(f"Max Depth: {best_config['max_depth']}")
print(f"Min Items: {best_config['min_items']}")
print(f"Best RMSE (Validation Set): {best_rmse}")
print(f"Best R2 (Validation Set): {best_r2}")
print()            
best_rmse = np.inf
best_r2 = -np.inf
best_config = {'max_depth': None, 'min_items': None}
reg = Regression()
for max_depth in max_depth_values:
    for min_items in min_items_values:
        # Train and validate the decision tree on the validation set
        _, rmse, r2,_  = reg.tree_regression(train_corr, train_y, valid_corr, valid_y, max_depth=max_depth, min_items=min_items)
        
        # If current model is better, update the best configuration
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_config['max_depth'] = max_depth
            best_config['min_items'] = min_items
print("Best Configuration:")
print(f"Max Depth: {best_config['max_depth']}")
print(f"Min Items: {best_config['min_items']}")
print(f"Best RMSE (Validation Set): {best_rmse}")
print(f"Best R2 (Validation Set): {best_r2}")
print()
best_rmse = np.inf
best_r2 = -np.inf
best_config = {'max_depth': None, 'min_items': None}
reg = Regression()
for max_depth in max_depth_values:
    for min_items in min_items_values:
        # Train and validate the decision tree on the validation set
        _, rmse, r2,_  = reg.tree_regression(train_lasso, train_y, valid_lasso, valid_y, max_depth=max_depth, min_items=min_items)
        
        # If current model is better, update the best configuration
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_config['max_depth'] = max_depth
            best_config['min_items'] = min_items
print("Best Configuration:")
print(f"Max Depth: {best_config['max_depth']}")
print(f"Min Items: {best_config['min_items']}")
print(f"Best RMSE (Validation Set): {best_rmse}")
print(f"Best R2 (Validation Set): {best_r2}")
print()

best_rmse = np.inf
best_r2 = -np.inf
best_config = {'max_depth': None, 'min_items': None}
reg = Regression()
for max_depth in max_depth_values:
    for min_items in min_items_values:
        # Train and validate the decision tree on the validation set
        _, rmse, r2,_ = reg.tree_regression(train_step, train_y, valid_step, valid_y, max_depth=max_depth, min_items=min_items)
        
        # If current model is better, update the best configuration
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_config['max_depth'] = max_depth
            best_config['min_items'] = min_items
print("Best Configuration:")
print(f"Max Depth: {best_config['max_depth']}")
print(f"Min Items: {best_config['min_items']}")
print(f"Best RMSE (Validation Set): {best_rmse}")
print(f"Best R2 (Validation Set): {best_r2}")
print()
#%%

#%%
train = pd.read_csv('./energydata/energy_train.csv', parse_dates=['date'], dayfirst=True)
val = pd.read_csv('./energydata/energy_val.csv', parse_dates=['date'], dayfirst=True)
test = pd.read_csv('./energydata/energy_test.csv', parse_dates=['date'], dayfirst=True)

print(train.shape, val.shape)
combined_train = pd.concat([train, val], axis=0)
print(combined_train.shape)
combined_train_x_pd = combined_train.drop(['Appliances', 'date'], axis=1)
test_x_pd = test.drop(['Appliances', 'date'], axis=1)

combined_train_x, combined_train_y = combined_train_x_pd.to_numpy(), combined_train['Appliances'].values
test_x, test_y = test_x_pd.to_numpy(), test['Appliances'].values


tree_pred, tree_rmse, tree_r2, _ = reg.tree_regression(combined_train_x, combined_train_y, test_x, test_y, max_depth=3, min_items=1)
print(tree_pred)
print("Decision Tree Regression - RMSE:", tree_rmse, "R2:", tree_r2)
#%%

#%%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming you have a trained DecisionTreeRegressor instance called 'tree'
# Use the best hyperparameters found in 1h (e.g., max_depth and min_items)

# Train a decision tree with the best configuration
test_pred, rmse, r2, tree_model = reg.tree_regression(combined_train_x, combined_train_y, test_x, test_y, max_depth=3, min_items=1)

# Visualize the decision tree, limiting the plot to the top 3 levels
plt.figure(figsize=(16, 10))
plot_tree(tree_model, feature_names=feature_names, max_depth=3, filled=True, rounded=True)
plt.show()
#%%
