import _init_paths
from util.vecs import vecs_io
import matplotlib.pyplot as plt
import numpy as np


# cmap Accent, Blues, BrBG, BuGn, BuPu, CMRmap, Dark2, GnBu, Greens, Greys, OrRd, Oranges, PRGn, Paired, Pastel1,
# Pastel2, PiYG, PuBu, PuBuGn, PuOr, PuRd, Purples, RdBu RdGy, RdPu, RdYlBu, RdYlGn, Reds, Set1, Set2, Set3,
# Spectral, Wistia, YlGn, YlGnBu, YlOrBr, YlOrRd
def show_data(data):
    # X是二维数组, 维度为2
    # Y是一维数组, 代表各个分类的质心
    plt.xticks(())
    plt.yticks(())
    # 呈现未归集前的数据
    plt.scatter(data[:, 0], data[:, 1], s=2)
    plt.savefig('data.png')


def show_clustering(data, partition, method):
    plt.xticks(())
    plt.yticks(())
    # s 代表大小，c代表颜色(可以是一维数组表示标签也可以是RGB)
    plt.scatter(base[:, 0], base[:, 1], c=label, s=2, cmap='Accent')
    plt.savefig(method + '.png')


def show_answer(data, partition, single_query, method):
    plt.xticks(())
    plt.yticks(())
    # s 代表大小，c代表颜色(可以是一维数组表示标签也可以是RGB)
    plt.scatter(base[:, 0], base[:, 1], c=label, s=2, cmap='Accent')
    plt.scatter(single_query[0], single_query[1], c='blue', s=100, alpha=0.5)
    plt.savefig(method + '.png')
    plt.close()


if __name__ == '__main__':
    base = vecs_io.fvecs_read('data/dataset/normalsmall_10/base.fvecs')[0]
    query = vecs_io.fvecs_read('data/dataset/normalsmall_10/query.fvecs')[0]

    show_data(base)

    method = 'knn'
    small_idx = 17
    large_idx = 0

    label = np.loadtxt('data/train_para/normalsmall_16_nn_1_%s_/Classifier_0/dataset_partition/partition.txt' % method)
    show_clustering(base, label, '%s_1_1' % method)
    show_answer(base, label, query[small_idx], '%s_1_1_small_recall' % method)
    show_answer(base, label, query[large_idx], '%s_1_1_large_recall' % method)
