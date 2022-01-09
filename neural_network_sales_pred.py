import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

items = pd.read_csv('items.csv')


def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def SIA(y_pred, y_true, eps=100):
    error = np.abs(y_pred - y_true)

    res = np.mean(((error - eps) < 0) & (y_pred >= 0))
    return res


category_dict = items.groupby('category').agg(['mean'])['n_sold'].to_dict()['mean']
stf_dict = items.groupby('shop_time_feedback').agg(['mean'])['n_sold'].to_dict()['mean']
category = items['category'].map(category_dict)
shop_time_feedback = items['shop_time_feedback'].map(stf_dict)

numerical_feats = items[['avg_rating', 'n_loved', 'price', 'rate_1', 'rate_2', 'rate_3', 'rate_4', 'rate_5', 'rate_with_cmt',
       'rate_with_imgvid', 'shop_age', 'shop_follower', 'shop_n_product', 'shop_n_review', 'shop_rate_feedback']].to_numpy()
X = np.hstack([numerical_feats, category[:, None], shop_time_feedback[:, None]])
y = items['n_sold'].to_numpy()

normalizer = StandardScaler()
X = normalizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1000, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =1000, random_state=1)

num_features = X.shape[1]
num_epochs = 150
batch_size = 64
learning_rate = 1e-2

model = Sequential()
model.add(Dense(32, input_dim=num_features, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer='adam',
              loss = tf.keras.losses.mean_absolute_error,
              metrics=['mae'])

model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath="nn_model.h5",
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)

# history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[model_save])

# pd.DataFrame.from_dict(history.history).to_csv('history.csv',index=False)
history = pd.read_csv('history.csv')

plt.plot(range(num_epochs), history['mae'])
plt.plot(range(num_epochs), history['val_mae'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.load_weights("nn_model.h5")

print('Train_loss:', mae_loss(y_train, model.predict(X_train).squeeze()))
print('Val loss:', mae_loss(y_val, model.predict(X_val).squeeze()))
print('Test loss:', mae_loss(y_test, model.predict(X_test).squeeze()))

print('SIA train accuracy:', SIA(y_train, model.predict(X_train).squeeze(), 100))
print('SIA val accuracy:', SIA(y_val, model.predict(X_val).squeeze(), 100))
print('SIA test accuracy:', SIA(y_test, model.predict(X_test).squeeze(), 100))

y_pred = model.predict(X_test).squeeze()
predLR = pd.DataFrame(list(zip(y_test,y_pred)), columns  = ['Actual', 'Predicted'] )
sns.lmplot('Actual', 'Predicted', predLR)
plt.title('Neural Network')
plt.show()