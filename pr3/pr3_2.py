import pandas as pd
import numpy as np


def missing_val(df):
    for column in df.columns:
        missing = np.mean(df[column].isna() * 100)
        print(f" {column} : {round(missing, 1)}%")
    print("==============================================")
    return


def find_duplicates(df):
    a = 0
    for item in df.duplicated():
        if item:
            a += 1
    print("Дупликатов: ", a)
    df.drop_duplicates()
    return


def main():
    df = pd.read_csv("ECDCCases.csv")
    missing_val(df)
    df.drop(columns=["geoId", 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'], inplace=True)
    med = np.mean(df['popData2019'])
    df['popData2019'].fillna(med, inplace=True)
    df['countryterritoryCode'].fillna("other", inplace=True)
    missing_val(df)
    print(df.describe())

    find_duplicates(df)
    return


if __name__ == "__main__":
    main()
