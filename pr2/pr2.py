import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import umap
import time


def bar_grapf(df):
    fig = go.Figure()
    fig.add_bar(x=[j for j in range(len(df))], y=df['subscribers'],
                marker=dict(color=[i for i in df['subscribers']],
                            colorscale='Inferno', coloraxis="coloraxis",
                            line=dict(color='black', width=2)))
    fig.update_layout(title={'text': "Best youtubers statistic",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      title_font_size=20, height=700, autosize=True
                      )
    fig.update_xaxes(title="rank youtuber", title_font_size=16, tickangle=315, tickfont_size=14,
                     showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.update_yaxes(title="subscribers", title_font_size=16, tickfont_size=14,
                     showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.show()
    return


def circle_grapf(df):
    fig = go.Figure()
    low, mid, high = 0, 0, 0
    new_arr = []
    labels = ['subscribers < 35kk', 'subscribers < 80kk', 'subscribers > 80kk']
    for items in df["subscribers"]:
        if items < 35_000_000:
            low += 1
        elif items < 80_000_000:
            mid += 1
        else:
            high += 1
    new_arr.append(low)
    new_arr.append(mid)
    new_arr.append(high)
    fig.add_trace(go.Pie(values=new_arr, labels=labels, marker_line_width=2))
    fig.show()
    return


def line_grapf(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot([j for j in range(len(df))], df['highest_yearly_earnings'], marker='o',
             color='crimson', markerfacecolor='white', markeredgecolor='black', markersize=2)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Highest yearly earnings')
    ax1.grid(True, color='mistyrose', linewidth=2)
    ax2.plot([j for j in range(len(df))], df['subscribers'], marker='o',
             color='crimson', markerfacecolor='white', markeredgecolor='black', markersize=2)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Subscribers')
    ax2.grid(True, color='mistyrose', linewidth=2)
    ax3.plot([j for j in range(len(df))], df['video views'], marker='o',
             color='crimson', markerfacecolor='white', markeredgecolor='black', markersize=2)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Video views')
    ax3.grid(True, color='mistyrose', linewidth=2)
    plt.show()
    return


def tsne_grapf():
    start = time.time()
    X, y = load_digits(return_X_y=True)
    embed = TSNE(n_components=2, perplexity=30, random_state=123)
    X_embedded = embed.fit_transform(X)
    end = time.time()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=0.5)
    print("Время выполнения функции: ", end - start, "секунд")
    plt.show()
    return


def umap_grapf():
    start = time.time()
    X, y = load_digits(return_X_y=True)
    manifold = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=123).fit(X, y)
    X_reduced = manifold.transform(X)
    end = time.time()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=0.5)
    print("Время выполнения функции: ", end - start, "секунд")
    plt.show()
    return


def main():
    df_orig = pd.read_csv("youtube_stat.csv", encoding_errors="replace")
    df = df_orig.drop_duplicates(subset="subscribers")
    print(df.info())
    print(df.head())
    a = input("Какую диаграмму вывести?(1-столбчатая, 2-круговая, 3-линейная, 4-t-SNE, 5-UMAP): ")
    a = int(a)
    if a == 1:
        bar_grapf(df)
    elif a == 2:
        circle_grapf(df)
    elif a == 3:
        line_grapf(df)
    elif a == 4:
        tsne_grapf()
    elif a == 5:
        umap_grapf()
    else:
        print("Неправильное значение")
    return


if __name__ == "__main__":
    main()
