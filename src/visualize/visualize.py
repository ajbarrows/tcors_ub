import matplotlib.pyplot as plt
import seaborn as sns


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

    sns.catplot(
        data=df_plt,
        x='prp_change_bin',
        y="counts",
        hue = 'dose',
        kind="bar"
    )
    plt.xticks(rotation = 45)
    plt.xlabel("Binned Proportion of Baseline CPD")
    plt.ylabel("Subjects")
    plt.suptitle("Categorical Model of Prp. Change in CPD from Baseline", y = 1)
    plt.text(x=1, y=40, s= str(n_cat) + " bins")

def clf_plt(df):
    p = sns.catplot(
        data = df,
        x = 'model',
        y = 'f1',
        hue = 'label',
        kind = 'bar'
    )
    p.legend.set_title(None)
    plt.title("Classification Performance")

