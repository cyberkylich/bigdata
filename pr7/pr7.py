import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import catboost as cb


def f(x):
    return 5.373 + 0.812 * x


def stacking(df, x):
    x_datasets = []
    y_datasets = []
    y = f(x)
    for i in range(10):
        sampled_data = df.sample(frac=0.8, replace=True, random_state=i)
        x_datasets.append(np.array(sampled_data["exp(in months)"]))
        y_datasets.append(np.array(sampled_data["salary(in thousands)"]))
    for i in range(10):
        plt.scatter(x_datasets[i], y_datasets[i], c='green', s=3)
    plt.plot(x, y, '--', color='black')
    plt.show()
    return x_datasets, y_datasets, y


def bagging(x, y, x_datasets, y_datasets):
    models = []
    for i in range(10):
        model_tree = tree.DecisionTreeRegressor(max_depth=8, random_state=1)
        model_tree.fit(x_datasets[i].reshape(-1, 1), y_datasets[i])
        models.append(model_tree)
    y_pred = []
    for i in range(len(models)):
        y_pred.append(models[i].predict(x.reshape(-1, 1)))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    for i in range(10):
        plt.scatter(x, y_pred[i], c='red', s=2)
    plt.plot(x, y, color='black')
    plt.show()

    mean_pred = np.array(y_pred).mean(axis=0)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.scatter(x, mean_pred, c='green', zorder=2)
    plt.plot(x, y, '--', color='black', lw=1)
    plt.show()

    model_tree = tree.DecisionTreeRegressor(max_depth=8, random_state=1)
    one_model = model_tree.fit(np.array(x_datasets).reshape(-1, 1), np.array(y_datasets).reshape(-1, 1))
    one_pred = one_model.predict(x.reshape(-1, 1))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.scatter(x, one_pred, c='green', zorder=2)
    plt.plot(x, y, '--', color='black', lw=1.5)
    plt.show()
    print('Коэффициент детерминации для случайного леса: ', r2_score(mean_pred, y))
    print('Коэффициент детерминации для одного дерева решений: ', r2_score(one_pred, y))
    return


def boosting(df):
    predictors = np.array(df['exp(in months)'])
    target = np.array(df['salary(in thousands)'])
    x_train, x_test, y_train, y_test = train_test_split(predictors.reshape(-1, 1), target, test_size=0.2, random_state=0)
    print('Размер для признаков обучающей выборки', x_train.shape, '\n',
          'Размер для признаков тестовой выборки', x_test.shape, '\n')
    random_forest = RandomForestRegressor()
    params_grid = {
        "max_depth": [12, 18],
        "min_samples_leaf": [3, 10],
        "min_samples_split": [6, 12]
    }

    grid_search_random_forest = GridSearchCV(estimator=random_forest, param_grid=params_grid,
                                             scoring="neg_mean_squared_error", cv=2)
    grid_search_random_forest.fit(x_train, y_train)
    best_model = grid_search_random_forest.best_estimator_
    y_preds_train = best_model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_preds_train)
    print('Mean Squared Error на тренировочных данных:', mse_train)
    y_preds_test = best_model.predict(x_test)
    mse_test = mean_squared_error(y_test, y_preds_test)
    print('Mean Squared Error на тестовых данных:', mse_test)

    model_catboost_clf = cb.CatBoostClassifier(iterations=100,
                                               task_type="GPU",
                                               devices='0')
    model_catboost_clf.fit(x_train, y_train)
    y_preds_t = model_catboost_clf.predict(x_train, task_type="CPU")
    mse_train = mean_squared_error(y_train, y_preds_t)
    print('Mean Squared Error на тренировочных данных:', mse_train)
    y_preds_tt = model_catboost_clf.predict(x_test, task_type="CPU")
    mse_test = mean_squared_error(y_test, y_preds_tt)
    print('Mean Squared Error на тестовых данных:', mse_test)
    return


def main():
    df = pd.read_csv('Experience-Salary.csv')
    x = np.array(df['exp(in months)'])
    y = np.array(df['salary(in thousands)'])
    print(df.info())
    x_datasets, y_datasets, y = stacking(df, x)
    bagging(x, y, x_datasets, y_datasets)
    boosting(df)
    return


if __name__ == "__main__":
    main()