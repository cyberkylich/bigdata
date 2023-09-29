import pandas as pd




def main():
    df = pd.read_csv("insurance.csv")
    print(df.describe())
    return


if __name__ == "__main__":
    main()