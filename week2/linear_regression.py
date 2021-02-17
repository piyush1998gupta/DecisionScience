import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def linear_Regression(x,y):
    total_rows =  x.size
    # training data set 70
    training_size = (70 * x.size)//100
    testing_size = total_rows - training_size;
    x_train = x[:training_size] # independent

    y_train = y[:training_size] # dependent


    #coefficients  w0 - intercept and w1 - slope or how effective that attribute is on predicting y

    # w1 = (sum(x*y) - mean(y)*sum(x) ) / (sum(x**2)  - mean(x)*sum(x))
    # w0 = mean(y) + w1* mean(x)

    w1 = ( (x_train*y_train).sum() - y_train.mean() * x_train.sum() ) / ( (x_train*x_train).sum() - x_train.mean() * x_train.sum())
    w0 = y_train.mean() - w1 * x_train.mean()
    print(w0," " ,w1)
    x_test = x[training_size:]
    y_actual = y[training_size:]
    y_observed = w0 + (w1*x_test)


    mean_error = ((y_observed-y_actual)**2).mean()
    before_error = ((y_train -(w0 + (w1 * x_train)) )**2).mean()
    print("Training  error " ,before_error)

    # plt.scatter(x_test,y_actual)
    # plt.plot(x_test,y_observed)
    # plt.show()



    return mean_error

df= pd.read_csv("advertising.csv")
tv = df["TV"].to_numpy()
radio = df["Radio"].to_numpy()
newspaper = df["Newspaper"].to_numpy()
sales = df["Sales"].to_numpy()


print(linear_Regression(tv,sales))

# print(linear_Regression(radio,sales))
# print(linear_Regression(newspaper,sales))