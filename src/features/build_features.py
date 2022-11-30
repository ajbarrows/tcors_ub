import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# functions 

def bin_cpd(df, plot = True):
    '''Cut total_cpd into evenly-spaced bins of 5 cpd, stopping at an upper limit of
    80 cpd (i.e., final bin is 80 - Inf).'''

    w12 = df[df["week"] == "week12"]
    print("Min CPD:" + str(min(df["total_cpd"])))
    print("Max CPD:" + str(max(df["total_cpd"])))

     # specify bins
    upper_limit = 80
    step = 5
    bins = np.arange(0, upper_limit + step, step=step)
    bins = np.append(bins, np.inf)

    if plot:
        # make histogram
        sns.histplot(
            data = w12,
            x = "total_cpd",
            bins = bins
        )
        plt.title("Week 12 CPD Distribution")
        plt.xlabel("Total CPD")
        plt.ylabel("Subjects")
        plt.savefig("../../reports/figures/w12cpd_hist.png", dpi = 300)

    # construct variable based on total_cpd bins
    df["total_cpd_bin"] = pd.cut(df["total_cpd"], bins, include_lowest=True).astype("category")
    df['cpd_bin_label'] = df.total_cpd_bin.cat.codes

    return(df)

def prp_change(df):
    '''Calculate proportion change in smoking from baseline using total experimental CPD.
    If experimental CPD is 0, coerce proportion change to -1.'''

    df.loc[df['week'] == 'week0', 'total_cpd'] = np.nan
    df['prp_change'] = np.where(df['total_cpd'] == 0, -1, df['total_cpd']/df['baseline_cpd'])
    df['prp_change'] = np.where(df['prp_change'] == np.Inf, 0, df['prp_change'])

    return(df)

def bin_prp_change(df, var_name):
    upper_limit = 2
    step = .1
    bins = np.arange(-1, upper_limit + step, step=step)
    bins = np.append(bins, np.inf)

     # construct variable based on prop_change
    df[var_name + "_bin"] = pd.cut(df[var_name], bins, include_lowest=True).astype("category")
    df[var_name + "_bin_label"] = df[var_name + "_bin"].cat.codes

    return df

def write_csv(df):
    df['prp_change_bin'] = df['prp_change_bin'].astype("str")
    df.to_csv("../../data/processed/model_features.csv")

# main

df = pd.read_csv("../../data/clean/s2_full.csv")
# df = bin_cpd(df, plot=False)
df = prp_change(df)
df = bin_prp_change(df, 'prp_change')

print(df.head())

df.to_pickle("../../data/processed/model_features.pkl")

write_csv(df)


