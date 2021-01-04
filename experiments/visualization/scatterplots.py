import matplotlib.pyplot as plt
import seaborn as sns


def basic_scatter(projection):
    x = projection[:, 0]
    y = projection[:, 1]
    sns.scatterplot(x, y)
    plt.show()
    plt.clf()
