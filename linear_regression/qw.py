from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from scipy.interpolate import spline

data = pd.read_csv('data.csv', index_col='#', encoding="cp1252")

y_ = np.array(data['a. ã/êÂò*÷'])
x_ = np.array(data['Píàãð/Píîì'])


x_plot = np.linspace(x_.min(), x_.max(), 3000)
y_plot = spline(x_, y_, x_plot)

def f(x):
    return x**6-x**5+x**4-x**3+x**2-x

f = np.vectorize(f)

y = f(x_)
print(y)

def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=100, input_dim=784, activation='relu'))
    model.add(Dense(output_dim=100, activation='softmax'))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

model = baseline_model()
model.fit(x_plot, y, nb_epoch=400, verbose = 0)

sb.set(style="white")

#plt.scatter(x_plot, y_plot, color='black')
plt.plot(x_plot, model.predict(x_plot), color='magenta', linewidth=2)
plt.show()
