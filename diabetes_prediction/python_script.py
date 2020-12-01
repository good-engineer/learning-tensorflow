# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load data
dataset = loadtxt('dataset.csv', delimiter=',')
X_train = dataset[0:615, 0:8]
y_train = dataset[0:615, 8]
# use 20 percent of dataset for evaluation
X_test = dataset[615:, 0:8]
y_test = dataset[615:, 8]
# define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# %%
# fit the keras model on dataset
model.fit(X_train, y_train, epochs=240, batch_size=15, verbose=0)


# %%
# make class predictions with the model
predictions = model.predict_classes(X_test, verbose=0)
for i in range(10):
    print('%s => %d (expected %d)' %
          (X_test[i].tolist(), predictions[i], y_test[i]))


# %%
# evaluate the model
_, accurancy_test = model.evaluate(X_test, y_test, verbose=0)
print('Accurancy: %.2f' % (accurancy*100))
