import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import umap


def kmeans(df):
    models = []
    score1 = []
    score2 = []
    for i in range(2, 10):
        model = KMeans(n_clusters=i, random_state=123, init='k-means++').fit(df)
        models.append(model)
        score1.append(model.inertia_)
        score2.append(silhouette_score(df, model.labels_))
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(np.arange(2, 10), score1, marker='o')
    ax[1].plot(np.arange(2, 10), score2, marker='o')
    ax[0].set_title('Функция стоимости')
    ax[1].set_title('Коэффициент силуэта')
    plt.show()

    model1 = KMeans(n_clusters=3, random_state=123, init='k-means++')
    model1.fit(df)
    df['Claster'] = model1.labels_
    # standard_embedding = umap.UMAP(random_state=42).fit_transform(df)
    # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=df['Claster'])
    plt.scatter(df['x'], df['y'], c=df['Claster'])
    plt.show()
    return


def hac(df):
    model2 = AgglomerativeClustering(3, compute_distances=True)
    clastering = model2.fit(df)
    df['Claster'] = clastering.labels_
    standard_embedding = umap.UMAP(random_state=42).fit_transform(df)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=df['Claster'])
    # plt.scatter(df['x'], df['y'], c=df['Claster'])
    plt.show()
    return


def dbscan(df):
    model3 = DBSCAN(eps=10, min_samples=10).fit(df)
    df['Claster'] = model3.labels_
    # standard_embedding = umap.UMAP(random_state=42).fit_transform(df)
    # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=df['Claster'])
    plt.scatter(df['x'], df['y'], c=df['Claster'])
    plt.show()
    return


def main():
    df = pd.read_csv("basic5.csv")
    print(df.info())
    # kmeans(df)
    # hac(df)
    dbscan(df)
    return


if __name__ == "__main__":
    main()
