import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def plot_path(df, deadline=False):
    df = df.sort_values(by=['trial', 't'])
    grid = sns.FacetGrid(df, col="trial", hue="trial", col_wrap=5, size=2.0, aspect=1.3)
    grid.map(plt.axhline, y=0, ls=":", c=".5")
    grid.map(plt.plot, "t", "reward", marker="o", ms=3, lw=.8)

    if deadline:
        grid.set(xticks=np.linspace(0, 50, 6, endpoint=True), xlim=(-1, 50), ylim=(-3, 13))
    else:
        grid.set(xticks=np.linspace(0, 150, 6, endpoint=True), xlim=(-1, 150), ylim=(-3, 13))
    grid.fig.tight_layout(w_pad=1)


def plot_results(df, xs):
    """

    :param df:
    :type df:
    :param xs:
    :type xs:
    :return:
    :rtype:
    """
    g = sns.PairGrid(df.sort_values("trial", ascending=False),
                     x_vars=xs, y_vars=["trial"],
                     size=10, aspect=.3)

    # Draw a dot plot using the stripplot function
    g.map(sns.stripplot, size=10, orient="h",
          palette="Blues", edgecolor="gray")

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(1, 13), xlabel="Rewards", ylabel="")

    # Use semantically meaningful titles for the columns
    titles = ["Random", "Basic agent", "Q-learning (basic)", "Q-learning (optimized)"]

    for ax, title in zip(g.axes.flat, titles):
        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)


def plot_opt(data, starting_point):
    """

    :param data:
    :type data:
    :param starting_point:
    :type starting_point:
    :return:
    :rtype:
    """
    n = 100 - starting_point
    data = data.sort_values(by=['trial', 't'])
    for i in data.initial_value.unique():
        plt.figure(figsize=(12, 8))
        print("initial_value = {}".format(i))
        df = data[data['initial_value'] == i].iloc[starting_point:, :]
        display(df.groupby(['learning_rate', 'discount_rate'])['success'].agg(np.sum).unstack()/n)
        sns.heatmap(df.groupby(['learning_rate', 'discount_rate'])['success'].agg(np.sum).unstack()/n, annot=True)
        plt.show()
