import numpy as np
import pandas as pd
import math
import talib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import ParamterGrid
from pprint import pprint
from sklearn.preprocessing import scale

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import tensowflow as tf
from keras.layers import Dense, Dropout

print(feature_names)
print(feature_names[:-4])
train_features = train_features.iloc[:,:-4]
test_features = test_features.iloc[:,:-4]
sc = scale()
scaled_train_features = sc.fit_transform(train_features)
scaled_test_features = sc.transform(test_features)

# create figure and list containing axes
f,ax = plt.subplots(nrows = 2,ncols = 1)
# plot histograms of before and after scaling
train_features.iloc[:,2].hist(ax = ax[0])
ax[1].hist(scaled_train_features[:,2])
plt.show()



##########################################

model = Sequential()
model.add(Dense(50,
	input_dim = scaled_train_features.shape[1],
	activation ='relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse')
history = model.fit(scaled_train_features, 
					train_targets, epochs = 50)

plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1],6)))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# calculate R^2 score
train_preds = model.predict(scaled_train_features)
print(r2_score(train_targets, train_preds))
plt.scatter(train_preds, train_targets)
plt.xlabel('predictions')
plt.ylabel('actual')
plt.show()

##########################################

# create loss function
def mean_squared_error(y_true, y_pred):
	loss = tf.square(y_true - y_pred)
	return tf.reduce_mean(loss, axis = -1)

# enable use of loss with keras
import keras.losses
keras.losses.mean_squared_error = mean_squared_error

# fit the model with our mse loss function
model.compile(optimizer = 'adam', loss = mean_squared_error)
history = model.fit(scaled_train_features, train_targets, epochs = 50)
tf.less(y_true*y_pred,0)

# create loss function

def sign_penalty(y_true, y_pred):
	penalty = 100
	loss = tf.where(tf.less(y_true*y_pred,0),
		penalty*tf.square(y_true-y_pred),
		tf.square(y_true - y_pred))
	return tf.reduce_mean(loss,axis = -1)

# enable use of loss with keras
keras.losses.sign_penalty = sign_penalty

#create the model
model = Sequential()
model.add(Dense(50,
				input_dim = scaled_train_features.shape[1],
				activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))

# fit the model with our custom 'sign_penalty' loss function
model.compule(optimizer = 'adam', loss = sign_penalty)
history = model.fit(scaled_train_features, train_targets, epochs = 50)

train_preds = model.predict(scaled_train_features)
# scatter the predictions vs actual
plt.scatter(train_preds,train_targets)
plt.xlabel('predictions')
plt.ylabel('actual')
plt.show()

##########################################

model = Sequential()
model.add(Dense(500, input_dim = scaled_train_features.shape[1],
	activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))

# make predictions from 2 neural net models
test_pred1 = model_1.predict(scaled_test_features)
test_pred2 = model_2.predict(Scaled_test_features)

# horizontally stack predictions and take the average across rows
test_preds = np.mean(np.hstack((test_pred1, test_pred2)), axis = 1)

from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:,:-4]
test_features = test_features.iloc[:,:-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()

from sklearn.neighbors import KNeighborsRegressor

for n in range(2,13):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)
    
    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)
    
    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features,test_targets))
    print()  # prints a blank line

# Create the model with the best-performing n_neighbors of 5
knn = KNeighborsRegressor(5)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label = 'test')
plt.legend()
plt.show()

from keras.models import Sequential
from keras.layers import Dense

# Create the model
model_1 = Sequential()
model_1.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Fit the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=25)

# Plot the losses from the fit
plt.plot(history.history['loss'])

# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label = 'test')
plt.legend()
plt.show()

import keras.losses
import tensorflow as tf

# Create loss function
def sign_penalty(y_true,y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty * tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)

keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)

# Create the model
model_2 = Sequential()
model_2.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_2.add(Dense(20, activation='relu'))
model_2.add(Dense(1, activation='linear'))

# Fit the model with our custom 'sign_penalty' loss function
model_2.compile(optimizer='adam', loss=sign_penalty)
history = model_2.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

# Evaluate R^2 scores
train_preds = model_2.predict(scaled_train_features)
test_preds = model_2.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets,test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds,test_targets, label='test')  # plot test set
plt.legend(); plt.show()

from keras.layers import Dropout

# Create model with dropout
model_3 = Sequential()
model_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(20, activation='relu'))
model_3.add(Dense(1, activation='linear'))

# Fit model with mean squared error loss function
model_3.compile(optimizer='adam', loss='mse')
history = model_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-5:])

from sklearn.metrics import r2_score

# Evaluate the R^2 scores
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label = 'train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend(); plt.show()