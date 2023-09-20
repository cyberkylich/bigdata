import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go



def bar_grapf(df):
    # plt.plot()
    # plt.bar(df['rank'], df['subscribers'], marker=dict(color=[4, 5, 6], coloraxis="coloraxis"))
    # plt.xlabel("Рейтинг")
    # plt.ylabel("Подписчики")
    # plt.show()
    fig = go.Figure()
    fig.add_bar(x = df['rank'], y = df['subscribers'])
    fig.show()
    return


def main():
    df = pd.read_csv("youtube_stat.csv", encoding_errors="replace")
    # df.drop_duplicates(subset=['date'], iplace=True)
    print(df.info())
    print(df.head())
    bar_grapf(df)
    return

if __name__ == "__main__":
    main()


