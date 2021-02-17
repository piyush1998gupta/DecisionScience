import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def sse(x,y,model):
    val= np.sum((y - model.predict(x))**2)
    return val

def tse(x,y):
    val = np.sum((y-np.mean(y))**2)
    return val

def r_squared(sse,tse):
    return 1 - sse/tse


def all_errors(x,y,model):
    sse_value = sse(x, y, model)
    tse_value = tse(x,y)
    r_squared_value = r_squared(sse_value,tse_value)
    print("sse : ",sse_value);
    print("tse : ",tse_value)
    print("R^2 : ", r_squared_value)

def plot_residuals(x,y):
    plt.scatter(x,y)
    plt.show()

def linear_Data():
    np.random.seed(10)
    x = np.arange(20)
    y = [x*3 + np.random.rand(1)*4 for x in range(20)]
    x=x.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    all_errors(x,y,model)
    # best fitt line

    # plt.plot(x, model.predict(x))
    # plt.scatter(x, y)
    # plt.show()
    plt.plot(y,model.predict(x))
    plt.show()

    # residual plot
    plot_residuals(x,y-model.predict(x))

def non_linear_Data():
    np.random.seed(10)
    x = np.arange(20)
    y = [x ** 3 + np.random.rand(1) * 4 for x in range(20)]
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    all_errors(x, y, model)
    # best fitt line

    # plt.plot(x, model.predict(x))
    # plt.scatter(x, y)
    # plt.show()

    # residual plot

    plot_residuals(x, y - model.predict(x))

linear_Data()
# non_linear_Data()

def linear_data_outlier():
    np.random.seed(10)
    x = np.arange(20)
    y = [x * 3 + np.random.rand(1) * 4 for x in range(20)]
    x = x.reshape(-1, 1)
    y[5][0]=76

    model = LinearRegression()
    model.fit(x, y)
    all_errors(x, y, model)
    model_without = LinearRegression()
    model_without.fit(np.delete(x,5,0),np.delete(y,5,0))

    # best fitt line

    plt.plot(x, model.predict(x))
    plt.plot(np.delete(x,5,0), model_without.predict(np.delete(x,5,0)),'--')

    plt.scatter(x, y)
    plt.show()

    # residual plot
    # plot_residuals(x, y - model.predict(x))

# linear_data_outlier()

def linear_data_high_leverage():
    np.random.seed(10)
    x = np.arange(20)
    y = [x * 3 + np.random.rand(1) * 4 for x in range(20)]
    x = x.reshape(-1, 1)
    y[18][0] = 89
    y[19][0] = 90
    model = LinearRegression()
    model.fit(x, y)
    all_errors(x, y, model)
    model_without = LinearRegression()
    model_without.fit(np.delete(x,[18,19],0),np.delete(y,[18,19],0))

    # best fitt line

    plt.plot(x, model.predict(x))
    plt.plot(np.delete(x,[18,19],0), model_without.predict(np.delete(x,[18,19],0)) , '--')

    plt.scatter(x, y)
    plt.show()

    # residual plot
    # plot_residuals(x, y - model.predict(x))

# linear_data_high_leverage()

def Homoscedasticity():
    np.random.seed(5)
    x = np.arange(20)
    y_homo = [x * 2 + np.random.rand(1) for x in range(20)]
    y_hetero = [x * 2 + np.random.rand(1) * 2 * x for x in range(20)]
    x_reshape = x.reshape(-1, 1)
    linear_homo = LinearRegression()
    linear_homo.fit(x_reshape, y_homo)
    linear_hetero = LinearRegression()
    linear_hetero.fit(x_reshape, y_hetero)
    y_homo_pred = linear_homo.predict(x_reshape)
    y_hetero_pred = linear_hetero.predict(x_reshape)
    # all_errors(x_reshape,y_homo,linear_homo)
    plt.scatter(y_homo_pred,y_homo-y_homo_pred)
    # all_errors(x_reshape,y_hetero,linear_hetero)
    plt.scatter(y_hetero_pred,y_hetero-y_hetero_pred)
    plt.show()

# Homoscedasticity()