import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.interpolate import spline

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv('data.csv', index_col='#', encoding="cp1252")

# print(data)

y = np.array(data['a. ã/êÂò*÷'])
x = np.array(data['Píàãð/Píîì'])
print(y, x)

sb.set(style="white")

x_plot = np.linspace(x.min(),x.max(),100)
y_plot = spline(x,y,x_plot)

plt.plot(x_plot, y_plot, color='cornflowerblue', linewidth=2,
         label="ground truth")

#x = np.sort(x[:20])
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

for count, degree in enumerate([6, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot_1 = model.predict(X_plot)
    plt.plot(x_plot, y_plot_1, color='teal', linewidth=2,
             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()