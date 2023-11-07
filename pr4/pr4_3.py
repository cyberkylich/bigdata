import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    df = pd.read_csv("insurance.csv")
    print(df.info())
    regions = df.region.unique()
    print(regions)
    #
    # # scipy ANOVA test
    groups = df.groupby('region').groups
    southwest = df['bmi'][groups['southwest']]
    southeast = df['bmi'][groups['southeast']]
    northwest = df['bmi'][groups['northwest']]
    northeast = df['bmi'][groups['northeast']]
    # print(stats.f_oneway(southwest, southeast, northwest, northeast))
    #
    # # anova_lm ANOVA test
    # model = ols('bmi ~ region', data=df).fit()
    # anova_result = sm.stats.anova_lm(model, typ=2)
    # print(anova_result)

    # t-Student
    region_pairs = []
    for reg1 in range(3):
        for reg2 in range(reg1 + 1, 4):
            region_pairs.append((regions[reg1], regions[reg2]))

    for reg1, reg2 in region_pairs:
        print(reg1, reg2)
        print(stats.ttest_ind(df['bmi'][groups[reg1]], df['bmi'][groups[reg2]]))

    #post-hok
    tukey = pairwise_tukeyhsd(endog=df['bmi'], groups=df['region'], alpha=0.05)
    tukey.plot_simultaneous()
    tukey.summary()
    plt.show()
    return


if __name__ == "__main__":
    main()