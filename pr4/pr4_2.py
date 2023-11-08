import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def regression(df, epochs, learning_rate):
    x = np.array(df[['TV']])
    y = np.array(df['sales'])
    b0, b1 = 0, 0
    for i in range(epochs):
        f = b1 * x[:, 0] + b0
        gr_b1 = 2/len(x) * np.sum((y - f) * (-x[:, 0]))
        gr_b0 = 2/len(x) * np.sum(y - f) * (-1)
        b1 = b1 - learning_rate * gr_b1
        b0 = b0 - learning_rate * gr_b0
        mse = np.sum((y - f)**2)/len(x)
        print(f"Итерация : {i + 1}")
        print(f"Наклон {b1}| Сдвиг {b0}")
        print(f"MSE {mse}")
    #sklearn
    model = LinearRegression()
    model.fit(x, y)
    model_a = model.coef_[0]
    model_b = model.intercept_
    model_pred = model_a * x + model_b

    plt.scatter(x, y, marker='o', color="red")
    plt.plot(x, model_pred, linewidth=2, color='black', label=f'Модель sklearn = {model_a:.2f}x + {model_b:.2f}')
    plt.plot(x, f, '--g', linewidth=2, label=f'Вручную = {b1}x + {b0}')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


def main():
    df = pd.read_csv("Advertising.csv")
    print(df.info())
    print(df.corr())
    regression(df, 100000, 0.0000339)
    return


if __name__ == "__main__":
    main()