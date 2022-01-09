import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import time

items = pd.read_csv('items.csv')


def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def SIA(y_pred, y_true, eps=100):
    error = np.abs(y_pred - y_true)

    res = np.mean(((error - eps) < 0) & (y_pred >= 0))
    return res


# mean encoding with categorical feature
print('All features and mean encoding with categorical feature')
category_dict = items.groupby('category').agg(['mean'])['n_sold'].to_dict()['mean']
stf_dict = items.groupby('shop_time_feedback').agg(['mean'])['n_sold'].to_dict()['mean']
category = items['category'].map(category_dict)
shop_time_feedback = items['shop_time_feedback'].map(stf_dict)

numerical_feats = items[['avg_rating', 'n_loved', 'price', 'rate_1', 'rate_2', 'rate_3', 'rate_4', 'rate_5', 'rate_with_cmt',
       'rate_with_imgvid', 'shop_age', 'shop_follower', 'shop_n_product', 'shop_n_review', 'shop_rate_feedback']].to_numpy()
X = np.hstack([numerical_feats, category[:, None], shop_time_feedback[:, None]])
y = items['n_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =1000, random_state=1)

forest = RandomForestRegressor(n_estimators=100, random_state=1)
st_before = time.time()
forest.fit(X_train, y_train)
et_before = time.time()
print('Time running before:', et_before - st_before)

print('Train loss: ', mae_loss(y_train, forest.predict(X_train)))
print('Val loss: ', mae_loss(y_val, forest.predict(X_val)))
print('Test loss: ', mae_loss(y_test, forest.predict(X_test)))

print('SIA train accuracy', SIA(y_train, forest.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, forest.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, forest.predict(X_test), 100))

features_list = ['avg_rating', 'n_loved', 'price', 'rate_1', 'rate_2', 'rate_3', 'rate_4', 'rate_5', 'rate_with_cmt', 'rate_with_imgvid', \
                 'shop_age', 'shop_follower', 'shop_n_product', 'shop_n_review', 'shop_rate_feedback', 'category', 'shop_time_feedback']

fig = plt.figure(figsize=(16, 10))
plt.barh(range(len(features_list)), forest.feature_importances_)
plt.yticks(range(len(features_list)), features_list)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

features_importance = list(zip(features_list, forest.feature_importances_))
features_importance = sorted(features_importance, key=lambda x: x[1], reverse=True)

sorted_features = [feat[0] for feat in features_importance]
sorted_importances = [feat_importance[1] for feat_importance in features_importance]
cumulative_values = np.cumsum(sorted_importances)

plt.plot(range(len(features_list)), cumulative_values, 'g-')
plt.hlines(y = 0.95, xmin=0, xmax=len(features_list), color = 'r', linestyles = 'dashed')
plt.xticks(range(len(features_list)), sorted_features, rotation = 'vertical')
plt.xlabel('Feature')
plt.ylabel('Cumulative importance')
plt.show()

n_important_feats = np.where(cumulative_values > 0.95)[0][0] + 1
important_feats = sorted_features[:n_important_feats]
feats_indice = [features_list.index(feat) for feat in important_feats]

print('Importance features with mean encoding categorical feature')

X_train = X_train[:, feats_indice]
X_val = X_val[:, feats_indice]
X_test = X_test[:, feats_indice]
forest = RandomForestRegressor(n_estimators=100, random_state=1)
st_after = time.time()
forest.fit(X_train, y_train)
et_after = time.time()
print('Time running after', et_after - st_after)

print('Train loss: ', mae_loss(y_train, forest.predict(X_train)))
print('Val loss: ', mae_loss(y_val, forest.predict(X_val)))
print('Test loss: ', mae_loss(y_test, forest.predict(X_test)))

print('SIA train accuracy', SIA(y_train, forest.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, forest.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, forest.predict(X_test), 100))

n_estimators = [int(x) for x in np.linspace(100, 1000, 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators' : n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

'''rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, random_state=1, n_iter=100, cv=4, verbose=2, n_jobs=-1)

rf_random.fit(X_train, y_train)
print(rf_random.best_params_)'''

param_grid = {
    'bootstrap': [False],
    'max_depth': [80, 90, 100, None],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 3],
    'n_estimators': [500, 600, 700]
}

'''rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 4, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)'''

print('After tuning hyper parameters')
rf = RandomForestRegressor(n_estimators=500, max_depth=100, max_features=2, min_samples_leaf=1, min_samples_split=2, bootstrap=True)
rf.fit(X_train, y_train)

print('Train loss: ', mae_loss(y_train, forest.predict(X_train)))
print('Val loss: ', mae_loss(y_val, forest.predict(X_val)))
print('Test loss: ', mae_loss(y_test, forest.predict(X_test)))

print('SIA train accuracy', SIA(y_train, forest.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, forest.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, forest.predict(X_test), 100))