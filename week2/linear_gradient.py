import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def grad_des(x,y):
    size = (70 * x.size)//100
    # size=x.size
    x_t = x[:size]

    y_t = y[:size]
    # print(x_t)
    # print(y_t)
    w0 ,w1 , l_a = 0,0,0.000035
    it=100

    # plt.scatter(x_t,y_t)
    for i in range(it):
        y_p = w0+w1*x_t
        error =  sum( [ val**2 for val in (y_p-y_t)] )/ size
        w0 = w0 - l_a * ( (-2 * sum(y_t - y_p)) / size )
        w1 = w1 - l_a * ( (-2* sum(x_t * (y_t - y_p))) / size )
        print(f"w0 :  {w0} , w1 : {w1} , error : {error} , alpha : {l_a},it : {i}")
        # plt.plot(x_t,y_p)


df = pd.read_csv("advertising.csv")
tv = df["TV"].to_numpy()
radio = df["Radio"].to_numpy()
newspaper = df["Newspaper"].to_numpy()
sales = df["Sales"].to_numpy()
# grad_des(tv,sales)
# plt.show()

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
grad_des(x,y)
# x1=np.array([1,2,3,4,5])
# y1=np.array([5,7,9,11,13])
# grad_des(x1,y1)
