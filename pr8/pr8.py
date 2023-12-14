import time
import pandas as pd
import matplotlib.pyplot as plt
from apriori_python import apriori as apr1
from apyori import apriori as apr2
from efficient_apriori import apriori as apr3
from fpgrowth_py import fpgrowth


def apriori1(df):
    transactions = []
    for i in range(df.shape[0]):
        row = df.iloc[i].dropna().tolist()
        transactions.append(row)
    t = []
    start = time.perf_counter()
    t1, rule = apr1(transactions, minSup=0.015, minConf=0.5)
    time1 = time.perf_counter()-start
    t.append(time1)
    print('=====apriori_python=====')
    for item in rule:
        print(item)

    start = time.perf_counter()
    rules = apr2(transactions=transactions, min_support=0.015, min_confidence=0.5, min_lift=1.0001)
    results = list(rules)
    time2 = (time.perf_counter()-start)
    t.append(time2)
    print('=====apriori_2=====')
    for result in results:
        for subset in result[2]:
            print(subset[0], subset[1])
            print("Support: {0}; Confidence: {1}; Lift: {2};".format(result[1], subset[2], subset[3]))
            print()

    start = time.perf_counter()
    itemsets, rules = apr3(transactions, min_support=0.015, min_confidence=0.5)
    time3 = time.perf_counter()-start
    t.append(time3)
    print('=====efficient_apriori=====')
    for i in range(len(rules)):
        print(rules[i])

    start = time.perf_counter()
    itemsets, rules = fpgrowth(transactions, minSupRatio=0.015, minConf=0.5)
    time4 = time.perf_counter()-start
    t.append(time4)
    print('=====fpgrowth=====')
    for i in range(len(rules)):
        print(rules[i])

    print("Время выполнения apriori_python", t[0], "\n")
    print("Время выполнения apriori 2", t[1], "\n")
    print("Время выполнения efficient_apriori", t[2], "\n")
    print("Время выполнения fpgrowth", t[3], "\n")
    plt.bar(["Apriori", "Apriori 2", "Efficient apriori", "fpgrowth"], t)
    plt.title("Время работы алгоритмов")
    plt.show()
    return


def main():
    df = pd.read_csv("data.csv")
    print(df.info())
    df.stack().value_counts(normalize=True)[0:20].plot(kind='bar', title='Относительная частота встречаемости')
    plt.show()
    df.stack().value_counts(normalize=True).apply(lambda item: item / df.shape[0])[0:20].plot(kind='bar'
                                                                       ,title='Фактическая частота встречаемости')
    plt.show()
    apriori1(df)
    return


if __name__ == "__main__":
    main()