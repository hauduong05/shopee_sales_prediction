import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

items = pd.read_csv('items.csv')


def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred, y_true))


def SIA(y_pred, y_true, eps=100):
    error = np.abs(y_pred - y_true)

    res = np.mean(((error - eps) < 0) & (y_pred >= 0))
    return res


# choose only numerical features
X = items[['avg_rating', 'n_loved', 'price', 'rate_1', 'rate_2', 'rate_3', 'rate_4', 'rate_5', 'rate_with_cmt',
           'rate_with_imgvid', \
           'shop_age', 'shop_follower', 'shop_n_product', 'shop_n_review', 'shop_rate_feedback']].to_numpy()
y = items['n_sold'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1000, random_state=1)

model_ln = LinearRegression()
model_ln.fit(X_train, y_train)

print('Train loss: ', mae_loss(y_train, model_ln.predict(X_train)))
print('Val loss: ', mae_loss(y_val, model_ln.predict(X_val)))
print('Test loss: ', mae_loss(y_test, model_ln.predict(X_test)))

print('SIA train accuracy', SIA(y_train, model_ln.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, model_ln.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, model_ln.predict(X_test), 100))

# numerical feature with l2 regularization
best_val = 1e9
best_alpha = -1
alphas = np.linspace(0.01, 10, 100)
model = None

for alpha in alphas:
    model_ridge = Ridge(alpha)
    model_ridge.fit(X_train, y_train)
    val_loss = mae_loss(y_val, model_ridge.predict(X_val))
    if val_loss < best_val:
        best_val = val_loss
        best_alpha = alpha
        model = model_ridge

print('Best val:', best_val)
print('l2 regularization')

print('Train loss: ', mae_loss(y_train, model.predict(X_train)))
print('Val loss: ', mae_loss(y_val, model.predict(X_val)))
print('Test loss: ', mae_loss(y_test, model.predict(X_test)))

print('SIA train accuracy', SIA(y_train, model.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, model.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, model.predict(X_test), 100))

# one hot encoding with categorical feature
shop_time_feedback_ohe = pd.get_dummies(items['shop_time_feedback'])
category_ohe = pd.get_dummies(items['category'])
X_ohe = np.hstack([X, shop_time_feedback_ohe, category_ohe])

X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size = 1000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =1000, random_state=1)

model_ln = LinearRegression()
model_ln.fit(X_train, y_train)

print('One hot encoding with categorical feature')
print('Train loss: ', mae_loss(y_train, model_ln.predict(X_train)))
print('Val loss: ', mae_loss(y_val, model_ln.predict(X_val)))
print('Test loss: ', mae_loss(y_test, model_ln.predict(X_test)))

print('SIA train accuracy', SIA(y_train, model_ln.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, model_ln.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, model_ln.predict(X_test), 100))

# mean encoding with categorical feature
category_dict = items.groupby("category").agg(["mean"])["n_sold"].to_dict()['mean']
shop_time_dict = items.groupby('shop_time_feedback').agg(['mean'])['n_sold'].to_dict()['mean']
category_me = items['category'].map(category_dict)
shop_time_feedback_me = items['shop_time_feedback'].map(shop_time_dict)

X_me = np.hstack([X, category_me[:, None], shop_time_feedback_me[:, None]])
X_train, X_test, y_train, y_test = train_test_split(X_me, y, test_size = 1000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =1000, random_state=1)

model_ln = LinearRegression()
model_ln.fit(X_train, y_train)

print('Mean encoding with categorical feature')

print('Train loss: ', mae_loss(y_train, model_ln.predict(X_train)))
print('Val loss: ', mae_loss(y_val, model_ln.predict(X_val)))
print('Test loss: ', mae_loss(y_test, model_ln.predict(X_test)))

print('SIA train accuracy', SIA(y_train, model_ln.predict(X_train), 100))
print('SIA val accuracy', SIA(y_val, model_ln.predict(X_val), 100))
print('SIA test accuracy', SIA(y_test, model_ln.predict(X_test), 100))