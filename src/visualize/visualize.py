import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_pointplot(df, yvar, ylabel, color_var, legend_title, title, filename, save=False):

    sns.pointplot(
        x = 'week',
        y = yvar,
        hue = color_var,
        data = df[df['week'] != 'week0']
        
    )
    plt.xticks(rotation = 90)
    plt.xlabel(None)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title = legend_title)
    plt.tight_layout()
    
    if save:
        fname = "../../reports/figures/" + filename
        plt.savefig(fname, dpi = 200)
    
    plt.show()


def plot_bins(df):

    n_cat = df['prp_change_bin'].nunique()
    df_plt = df.groupby(['dose', 'prp_change_bin'])['prp_change_bin'].size().reset_index(name = "counts")
    fig = sns.catplot(
        data=df_plt,
        x='prp_change_bin',
        y="counts",
        hue = 'dose',
        kind="bar",
        height = 8,
        aspect = 11/8
    )

    sns.move_legend(    
        fig, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    plt.xticks(rotation = 45)
    plt.xlabel("Binned Proportion of Baseline CPD")
    plt.ylabel("Subjects")
    plt.text(x=1, y=20, s= str(n_cat) + " bins")




def clf_plt(df):
    p = sns.catplot(
        data = df,
        x = 'model',
        y = 'f1',
        hue = 'label',
        kind = 'bar'
    )
    p.legend.set_title(None)
    plt.title("Classification Performance: 'Lockbox' Set")

def rmse_cv(rmse_df, title=None):
    # rmse_df = pd.DataFrame(rmse_dict)
    rmse_df = pd.melt(rmse_df, id_vars='model', var_name='fold', value_name='rmse')
    rmse_df['epoch'] = rmse_df.groupby('fold').cumcount()

    sns.lineplot(
        data=rmse_df,
        x = "epoch",
        y = "rmse",
        hue = "fold",
        col = 'model'
    )
    plt.title(title)
