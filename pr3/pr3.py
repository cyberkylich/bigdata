import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
import seaborn as sns

def bar_grapf(df):
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].hist(df['age'], bins=10, edgecolor='black', color='blue', label='bins = 8')
    ax[0].legend()
    ax[1].hist(df['bmi'], bins=30, edgecolor='black', color='blue', label='bins = 30')
    ax[1].legend()
    ax[2].hist(df['children'], bins=5, edgecolor='black', color='blue', label='bins = 5')
    ax[2].legend()
    ax[3].hist(df['charges'], bins=15, edgecolor='black', color='blue', label='bins = 15')
    ax[3].legend()
    mean1, mean2 = np.mean(df['bmi']), np.mean(df['charges'])
    moda1, moda2 = sts.mode(df['bmi']), sts.mode(df['charges'])
    med1, med2 = np.median(df['bmi']), np.median(df['charges'])
    std1, std2 = df['bmi'].std(), df['charges'].std()
    raz1, raz2 = df['bmi'].max() - df['bmi'].min(), df['charges'].max() - df['charges'].min()
    iqr1, iqr2 = sts.iqr(df['bmi'], interpolation='midpoint'), sts.iqr(df['charges'], interpolation='midpoint')
    print('Среднее bmi: ', mean1, 'Среднее charges: ', mean2)
    print('Мода bmi: ', moda1, 'Мода charges: ', moda2)
    print('Медиана bmi: ', med1, 'Медиана charges: ', med2)
    print('Стандартное отклонение bmi: ', std1, 'Стандартное отклонение charges: ', std2)
    print('Размах bmi: ', raz1, 'Размах charges: ', raz2)
    print('Межквартильный размах bmi: ', iqr1, 'Межквартильный размах charges: ', iqr2)
    fig, bx = plt.subplots(1, 2, figsize=(15, 4))
    bx[0].hist(df['bmi'], bins=30, edgecolor='black', color='blue', label='bins = 30')
    bx[0].axvline(mean1, color='red', linewidth=2, label='Среднее')
    bx[0].axvline(moda1[0], color='green', linewidth=2, label='Мода')
    bx[0].axvline(med1, color='yellow', linewidth=2, label='Медиана')
    bx[0].legend()
    bx[1].hist(df['charges'], bins=15, edgecolor='black', color='blue', label='bins = 15')
    bx[1].axvline(mean2, color='red', linewidth=2, label='Среднее')
    bx[1].axvline(moda2[0], color='green', linewidth=2, label='Мода')
    bx[1].axvline(med2, color='yellow', linewidth=2, label='Медиана')
    bx[1].legend()
    plt.show()
    return


def box_plot(df):
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].boxplot(df['age'], vert=False)
    ax[0].set_title('age')
    ax[1].boxplot(df['bmi'], vert=False)
    ax[1].set_title('bmi')
    ax[2].boxplot(df['children'], vert=False)
    ax[2].set_title('children')
    ax[3].boxplot(df['charges'], vert=False)
    ax[3].set_title('charges')
    plt.show()
    return


def theorem(df):
    num_samples = 300
    len_samples = [10, 50, 100]
    fig, ax = plt.subplots(len(len_samples), figsize=(30, 8))
    means_charges = []
    means_bmi = []
    distance95_charge = {}
    distance95_bmi = {}
    distance99_charge = {}
    distance99_bmi = {}
    for i in range(len(len_samples)):
        sample_means_charges = []
        sample_means_bmi = []
        for k in range(num_samples):
            sample_charges = np.random.choice(df['charges'], size=len_samples[i], replace=True)
            sample_mean_charges = np.mean(sample_charges)
            sample_means_charges.append(sample_mean_charges)
            sample_bmi = np.random.choice(df['bmi'], size=len_samples[i], replace=True)
            sample_mean_bmi = np.mean(sample_bmi)
            sample_means_bmi.append(sample_mean_bmi)
        means_charges.append(sample_means_charges)
        means_bmi.append(sample_means_bmi)
        ax[i].hist(sample_means_charges, bins=15, edgecolor='black')
        ax[i].set_title('sample len = ' + str(len_samples[i]))
        sample_means_charges = np.array(sample_means_charges)
        sample_means_bmi = np.array(sample_means_bmi)
        print("Длина выборки: ", len_samples[i])
        print("Стандартное отклонение charges: ", sample_means_charges.std())
        print("Среднее отклонение charges: ", np.mean(sample_means_charges))
        print("Стандартное отклонение bmi: ", sample_means_bmi.std())
        print("Среднее отклонение bmi: ", np.mean(sample_means_bmi))
        distance95_charge[len_samples[i]] = [
            np.mean(sample_means_charges) - 1.96 * sample_means_charges.std() / np.sqrt(len_samples[i]),
            np.mean(sample_means_charges) + 1.96 * sample_means_charges.std() / np.sqrt(len_samples[i])]
        distance95_bmi[len_samples[i]] = [
            np.mean(sample_means_bmi) - 1.96 * sample_means_bmi.std() / np.sqrt(len_samples[i]),
            np.mean(sample_means_bmi) + 1.96 * sample_means_bmi.std() / np.sqrt(len_samples[i])]
        distance99_charge[len_samples[i]] = [
            np.mean(sample_means_charges) - 2.58 * sample_means_charges.std() / np.sqrt(len_samples[i]),
            np.mean(sample_means_charges) + 2.58 * sample_means_charges.std() / np.sqrt(len_samples[i])]
        distance99_bmi[len_samples[i]] = [
            np.mean(sample_means_bmi) - 2.58 * sample_means_bmi.std() / np.sqrt(len_samples[i]),
            np.mean(sample_means_bmi) + 2.58 * sample_means_bmi.std() / np.sqrt(len_samples[i])]
    print("Доверительный интервал 95% для charge: ", distance95_charge)
    print("Доверительный интервал 95% для bmi: ", distance95_bmi)
    print("Доверительный интервал 99% для charge: ", distance99_charge)
    print("Доверительный интервал 99% для bmi: ", distance99_bmi)
    plt.show()
    return


def normal_distribution(df):
    normal_quantiles = np.random.normal(0, 1, 1338)
    x = np.sort(normal_quantiles)
    y_bmi = np.sort(df['bmi'])
    y_charges = np.sort(df['charges'])
    sns.jointplot(x=x, y=y_bmi, kind="reg", truncate=True, color="blue")
    sns.jointplot(x=x, y=y_charges, kind="reg", truncate=True, color="purple")
    test_bmi = (df['bmi'] - np.mean(df['bmi']))/df['bmi'].std()
    test_charges = (df['charges'] - np.mean(df['charges'])) / df['charges'].std()
    print("KS-тест для bmi: ", sts.kstest(test_bmi, 'norm'))
    print("KS-тест для charges: ", sts.kstest(test_charges, 'norm'))
    plt.show()
    return


def main():
    df = pd.read_csv("insurance.csv")
    print(df.info())
    # bar_grapf(df)
    # box_plot(df)
    # theorem(df)
    normal_distribution(df)
    return


if __name__ == "__main__":
    main()
