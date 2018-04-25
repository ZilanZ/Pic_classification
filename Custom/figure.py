import matplotlib.pyplot as plt
import numpy as np


def plot_figure(phase, loss, acc, prc, rec, f1, result_dir, class_names):
    prc = np.nan_to_num(prc)
    rec = np.nan_to_num(rec)
    f1 = np.nan_to_num(f1)

    plt.figure()

    ax1 = plt.subplot(221)
    ax1.set_title('Loss')

    ax2 = plt.subplot(222)
    ax2.set_title('Accuracy')

    ax3 = plt.subplot(223)
    ax3.set_title('Recall ~ Precision')

    ax4 = plt.subplot(224)
    ax4.set_title('F1 measure')

    x = range(len(acc))

    prc = list(map(list, zip(*prc)))
    rec = list(map(list, zip(*rec)))
    f1 = list(map(list, zip(*f1)))
    # Accuracy
    plt.sca(ax1)
    plt.plot(x, loss)
    plt.sca(ax2)
    plt.plot(x, acc)
    # Precision, Recall, F1
    for it in range(len(class_names)):
        plt.sca(ax3)
        plt.scatter(prc[it], rec[it], label=class_names[it])
        plt.sca(ax4)
        plt.plot(x, f1[it], label=class_names[it])
    ax3.legend(loc='lower right', fontsize='xx-small')
    ax4.legend(loc='lower right', fontsize='xx-small')

    fig = open(result_dir + '/plot_' + phase + '.pdf', 'w')
    plt.savefig(result_dir + '/plot_' + phase + '.pdf')
    fig.close()
