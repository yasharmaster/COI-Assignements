from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy
import keras

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
dataframe = pd.read_csv("tic-tac-toe.csv", delimiter=",", header=None)
dataset = dataframe.values

numpy.random.shuffle(dataset)

original_input = dataset[:,0:9]
original_output = dataset[:,9]

shape_of_input = original_input.shape
final_input = numpy.zeros(shape_of_input)

for x in range(0, shape_of_input[0]):
    for y in range(0, shape_of_input[1]):
        if original_input[x, y] == 'x':
            final_input[x, y] = 1
        elif original_input[x, y] == 'b':
            final_input[x, y] = 0
        elif original_input[x, y] == 'o':
            final_input[x, y] = -1

final_output = []
for w in original_output:
    if(w=="positive"):
        final_output.append(1)
    else:
        final_output.append(-1)

# print (final_output)
# print (final_input)

training_input = final_input[:858,:]
training_output = final_output[:858]

testing_input = final_input[858:,:]
testing_output = final_output[858:]

# print (testing_input)
# print (testing_output)

# create model
model = Sequential()
model.add(Dense(20, input_dim=9, init='uniform', activation='relu'))
model.add(Dense(25, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit the model
model.fit(training_input, training_output, nb_epoch=20, batch_size=10,  verbose=2)

scores = model.evaluate(training_input, training_output)
print ("\nTraining Accuracy: %.4f%%" %(scores[1]*100))

scores = model.evaluate(testing_input, testing_output)
print ("\nTesting Accuracy: %.4f%%" %(scores[1]*100))

err = []
err2=[]
e=[]
e2=[]
val=[]
for l in range(1,30):
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, init='uniform', activation='relu'))
    model.add(Dense(l, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Fit the model
    model.fit(training_input, training_output)
    scores = model.evaluate(training_input, training_output)
    print ("Training Accuracy: %.4f%%" %(scores[1]*100))
    val.append(l)
    err.append(scores[1]*100)
    e.append(scores[1]*100)
    scores = model.evaluate(testing_input, testing_output)
    print ("Testing Accuracy: %.4f%%" %(scores[1]*100))
    err2.append(scores[1]*100)
    e2.append(scores[1]*100)

import matplotlib.pyplot as plt
plt.plot(val,err2,'r',err,'g')
plt.ylabel('accuracy')
plt.xlabel('number of hidden layers')
plt.show()
plt.plot(val,e2,'r',e,'g')
plt.ylabel('loss')
plt.xlabel('number of hidden layers')
plt.show()

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


import time
times = []
val=[]
err=[]
e=[]
err2=[]
e2=[]
for l in range(1,30):
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    adam = keras.optimizers.Adam(lr=l/1000, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Fit the model
    time_callback = TimeHistory()
    model.fit(training_input, training_output,  nb_epoch=1, batch_size=1,  verbose=2,callbacks=[time_callback])
    times.append(time_callback.times)
    scores = model.evaluate(training_input, training_output)
    print ("Training Accuracy: %.4f%%" %(scores[1]*100))
    val.append(l)
    err.append(scores[1]*100)
    e.append(scores[0]*100)
    scores = model.evaluate(testing_input, testing_output)
    print ("Testing Accuracy: %.4f%%" %(scores[1]*100))
    err2.append(scores[1]*100)
    e2.append(scores[0]*100)

import matplotlib.pyplot as plt
plt.plot(val,err2,'r',err,'g')
plt.ylabel('accuracy')
plt.xlabel('learning rate')
plt.show()
plt.plot(val,e2,'r',e,'g')
plt.ylabel('loss')
plt.xlabel('learning rate')
plt.show()
a=[]
for e in err:
    a.append(100-e)
a2=[]
for e in err2:
    a2.append(100-e)
plt.plot(val,a2,'r',a,'g')
plt.ylabel('error')
plt.xlabel('learning rate')
plt.show()
plt.plot(val,times)
plt.ylabel('time to convergance')
plt.xlabel('learning rate')
plt.show()
plt.plot(times,a2,'r',a,'g')
plt.ylabel('time to convergance')
plt.xlabel('learning rate')
plt.show()


err = []
err2=[]
e=[]
e2=[]
val=[]
for l in range(1,200,10):
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, init='uniform', activation='relu'))
    model.add(Dense(l, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # Fit the model
    model.fit(training_input, training_output)
    scores = model.evaluate(training_input, training_output)
    print ("Accuracy: %.4f%%" %(scores[1]*100))
    val.append(l)
    err.append(scores[1]*100)
    e.append(scores[0]*100)
    scores = model.evaluate(testing_input, testing_output)
    print ("Accuracy: %.4f%%" %(scores[1]*100))
    err2.append(scores[1]*100)
    e2.append(scores[0]*100)


import matplotlib.pyplot as plt
plt.plot(val,e2,'r')
plt.plot(val,e)
plt.ylabel('loss')
plt.xlabel('number of hidden layers')
plt.show()
plt.plot(val,err2,'r')
plt.plot(val,err)
plt.ylabel('accuracy')
plt.xlabel('number of hidden layers')
plt.show()

import matplotlib.pyplot as plt
plt.plot(val,e2)
plt.plot(val,e)
plt.plot(val,err2)
plt.plot(val,err)
plt.show()