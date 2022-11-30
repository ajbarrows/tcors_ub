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