import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TEMPLATE = "{}.png"


def basic_scatter(projection):
    x = projection[:, 0]
    y = projection[:, 1]
    sns.scatterplot(x, y)
    plt.show()
    plt.clf()


def basic_scatter_labeled(projection, labels, scaling):
    x = projection[:, 0]
    y = projection[:, 1]
    sns.scatterplot(x, y, hue=labels).set_title(scaling)
    plt.savefig('{}.png'.format(scaling))
    plt.clf()

def movement_scatter(true, preds, targets, name):
    true = true.reshape((-1, 1, 2))
    preds = preds.reshape((-1, 1, 2))
    both = np.concatenate([true, preds], axis=1)
    fig, ax = plt.subplots()
    for entity in both:
        lines = ax.plot(entity[:, 0], entity[:, 1], '-o', markersize=3, markevery=[0])
        for line in lines:
            color = str(line.get_color())
            ax.plot(entity[1, 0], entity[1, 1], color=color, marker='s', markersize=3, markevery=[0])
        # ax.arrow(entity[1, 0], entity[1, 1], entity[1, 0], entity[1, 1], shape='full', lw=0, length_includes_head=True, head_width=.002)
        # ax.annotate(text='', xy=(entity[1, 0], entity[1, 1]), xytext=(0, 0))
    fig.set_size_inches(12, 12)
    fig.savefig(TEMPLATE.format(str(name)), format='png')
    plt.close()
