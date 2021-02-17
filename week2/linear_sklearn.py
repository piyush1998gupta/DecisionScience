import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd

def sklearn(x,y):

    size = (70 * x.size)//100
    # size=10
    x_t = x[:size,np.newaxis]
    # print(x_t)

    y_t = y[:size,np.newaxis]
    model = linear_model.LinearRegression()
    model.fit(x_t,y_t)

    y_p = model.predict(x[size:,np.newaxis])
    print("Mean squared error ", mean_squared_error(y[size:], y_p))
    print("weights ", model.coef_)
    print("intercept ", model.intercept_)
    print("score on training", model.score(x_t, y_t))
    print("score on testing", model.score(x[size:, np.newaxis], y[size:]))
    plt.scatter(x[size:],y[size:])
    plt.plot(x[size:],y_p)
    plt.show()


def sklearn_2d(x,y):
    print(x)
    size = (2 * x.shape[0])//100
    # size=10
    # print(size)
    x_t = x[:size]
    # print(x_t)

    y_t = y[:size]
    model = linear_model.LinearRegression()
    model.fit(x_t,y_t)

    y_p = model.predict(x[size:])
    print("training mean square error : ",mean_squared_error(y[:size],model.predict(x_t)))
    print(" testing Mean squared error ", mean_squared_error(y[size:], y_p))
    print("weights ", model.coef_)
    print("intercept ", model.intercept_)
    print("score on training", model.score(x_t, y_t))
    print("score on testing", model.score(x[size:], y[size:]))
    # plt.scatter(x[size:],y[size:])
    # plt.plot(x[size:],y_p)
    # plt.show()


df = pd.read_csv("advertising.csv")
tv = df["TV"].to_numpy()
radio = df["Radio"].to_numpy()
tv_2d = tv.reshape(-1,1)
radio_2d = radio.reshape(-1,1)
data = np.concatenate((tv_2d,radio_2d),axis=1)
# print(data[:23])
newspaper = df["Newspaper"].to_numpy()
sales = df["Sales"].to_numpy()
# sklearn(tv,sales)
print(data.shape[0])
sklearn_2d(data,sales)