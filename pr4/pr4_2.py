import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sm


def regression(df):
    x = df['TV']
    y = df['sales']
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum_xy = 0
    sum_xx = 0
    for i in range(x.count()):
        sum_xy += x[i]*y[i]
        sum_xx += x[i]**2
    ss_xy = sum_xy - x.count()*x_mean*y_mean
    ss_xx = sum_xx - x.count()*(x_mean**2)
    b1 = ss_xy / ss_xx
    b0 = y_mean - b1*x_mean
    f = [b0 + b1 * x for x in x]
    tt = 0
    for i in range(y.count()):
        tt += (y[i] - f[i])**2
    mse = tt / x.count()
    print("Коэффициент сдвига = ", b0)
    print("Угол наклона = ", b1)
    print("MSE = ", mse)
    plt.scatter(x, y, marker='o', color="red")
    plt.plot(x, f)
    plt.show()
    return


def main():
    df = pd.read_csv("Advertising.csv")
    print(df.info())
    print(df.corr())
    regression(df)

    return


if __name__ == "__main__":
    main()