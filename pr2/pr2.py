import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go


def bar_grapf(df):
    fig = go.Figure()
    fig.add_bar(x=[j for j in range(len(df))], y=df['subscribers'], marker=dict(color=[i for i in df['subscribers']],
                                                                                colorscale='Inferno',
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
    # low = []
    # mid = []
    # high = []
    # for items in df["subscribers"]:
    #     if items < 35_000_000:
    #         low.append(items)
    #     elif items < 80_000_000:
    #         mid.append(items)
    #     else:
    #         high.append(items)

    low = 0
    mid = 0
    high = 0
    new_arr = []
    labels = ['subscribers < 35 000 000', 'subscribers < 80 000 000', 'subscribers > 80 000 000']
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


def main():
    df_orig = pd.read_csv("youtube_stat.csv", encoding_errors="replace")
    df = df_orig.drop_duplicates(subset="subscribers")
    print(len(df))
    print(df.info())
    print(df.head())
    a = input("Какую диаграмму вывести?(1-столбчатая, 2-круговая, 3-линейная): ")
    a = int(a)
    if a == 1:
        bar_grapf(df)
    elif a == 2:
        circle_grapf(df)
    return


if __name__ == "__main__":
    main()
