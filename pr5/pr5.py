import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def logreg(df):
    predictors = df.drop('Pclass', axis=1)
    target = df['Pclass']
    x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size=0.9, shuffle=True)
    print('Размер для признаков обучающей выборки', x_train.shape, '\n',
          'Размер для признаков тестовой выборки', x_test.shape, '\n',
          'Размер для целевого показателя обучающей выборки', y_train.shape, '\n',
          'Размер для показателя тестовой выборки', y_test.shape)
    # Log regression
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print('Предсказанные значения : ', y_predict)
    print('Исходные значения : ', np.array(y_test))
    print('classification_report Log regression ', '\n',
          classification_report(y_test, y_predict))
    plt.rcParams['figure.figsize'] = (10, 10)
    fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto=True)
    fig.update_layout(xaxis_title='Target', yaxis_title='Prediction')
    fig.show()
    # SVM
    param_kernel = ('linear', 'rbf', 'poly', 'sigmoid')
    parameters = {'kernel': param_kernel}
    model_svc = SVC()
    grid_search_svm = GridSearchCV(estimator=model_svc, param_grid=parameters, cv=6)
    grid_search_svm.fit(x_train, y_train)
    best_model_svc = grid_search_svm.best_estimator_
    print(best_model_svc.kernel)
    svm_preds = best_model_svc.predict(x_test)
    print('Предсказанные значения : ', svm_preds)
    print('Исходные значения : ', np.array(y_test))
    print('classification_report SVM ', '\n',
          classification_report(svm_preds, y_test))
    fig = px.imshow(confusion_matrix(y_test, svm_preds), text_auto=True)
    fig.update_layout(xaxis_title='Target', yaxis_title='Prediction')
    fig.show()
    # KNN
    number_of_neighbors = np.arange(3, 10)
    model_knn = KNeighborsClassifier()
    params = {'n_neighbors': number_of_neighbors}
    grid_search = GridSearchCV(estimator=model_knn, param_grid=params, cv=6)
    grid_search.fit(x_train, y_train)
    print("best_score ", grid_search.best_score_)
    print("best num of neighbors ", grid_search.best_estimator_)
    knn_preds = grid_search.predict(x_test)
    print('Предсказанные значения : ', knn_preds)
    print('Исходные значения : ', np.array(y_test))
    print('classification_report KNN ', '\n',
          classification_report(knn_preds, y_test))
    fig = px.imshow(confusion_matrix(y_test, knn_preds), text_auto=True)
    fig.update_layout(xaxis_title='Target', yaxis_title='Prediction')
    fig.show()
    return


def main():
    df = pd.read_csv("Titanic.csv")
    df.drop('Sex', axis=1, inplace=True)
    print(df.info())
    print(df.head())
    plt.hist(df['Pclass'], bins=3, ec='black')
    plt.show()
    logreg(df)


if __name__ == "__main__":
    main()
