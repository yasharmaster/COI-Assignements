import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
dataframe = pd.read_csv("airfoil_self_noise.csv", delimiter=",", header=None)
dataset = dataframe.values

np.random.shuffle(dataset)

original_input = dataset[:,0:5]
original_output = dataset[:,5]

X_train = original_input[:1300,:]
Y_train = original_output[:1300]

X_test = original_input[1301:,:]
Y_test = original_output[1301:]

##define base model
def base_model():
     model = Sequential()
     model.add(Dense(20, input_dim=5, init='normal', activation='relu'))
     model.add(Dense(25, init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model

seed = 7
np.random.seed(seed)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5,verbose=0)
clf.fit(X_test,Y_test)


score = mean_absolute_error(Y_test, clf.predict(X_test))
print (score)

score = mean_absolute_error(Y_train, clf.predict(X_train))
print (score)


scores=[]
scores2=[]
v=[]
for i in range(1,50,1):
    ##define base model
    def base_model():
        model = Sequential()
        model.add(Dense(20, input_dim=5, init='normal', activation='relu'))
        model.add(Dense(i, init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        model.compile(loss='mean_squared_error', optimizer = 'adam')
        return model
    seed = 7
    np.random.seed(seed)
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5,verbose=0)
    clf.fit(X_test,Y_test)
    from sklearn.metrics import mean_absolute_error
    score = mean_absolute_error(Y_test, clf.predict(X_test))
    scores.append(score)
    v.append(i)
    score = mean_absolute_error(Y_train, clf.predict(X_train))
    scores2.append(score)


import matplotlib.pyplot as plt
plt.plot(v,scores,'g--')
plt.plot(v,scores2,'b')
plt.show()

import time
import keras
from sklearn.metrics import mean_absolute_error
times = []
scores=[]
scores2=[]
v=[]
for i in range(0,30):
    def base_model():
        model = Sequential()
        model.add(Dense(20, input_dim=5, init='normal', activation='relu'))
        model.add(Dense(25, init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        adam = keras.optimizers.Adam(lr=i/1000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer = adam)
        return model
    seed = 7
    np.random.seed(seed)
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    epoch_time_start = time.time()
    clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=5,verbose=0)
    clf.fit(X_test,Y_test)
    times.append(time.time() - epoch_time_start)
    score = mean_absolute_error(Y_test, clf.predict(X_test))
    scores.append(score)
    v.append(i)
    score = mean_absolute_error(Y_train, clf.predict(X_train))
    scores2.append(score)
    scores2

import matplotlib.pyplot as plt
plt.plot(v,scores,'g')
plt.plot(v,scores2,'b')
plt.ylabel('loss')
plt.xlabel('learning rate')
plt.show()
plt.plot(times,v,'g')
plt.ylabel('Learning rate')
plt.xlabel('time to convergance')
plt.show()
plt.plot(times,scores,'g')
plt.plot(times,scores2,'b')
plt.ylabel('Loss or error')
plt.xlabel('time to convergance')
plt.show()