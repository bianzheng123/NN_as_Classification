import numpy as np

n_classifier = 4
method = 'partition_knn'

result = np.loadtxt('data/result/normalsmall_4_nn_%d_%s_/recall_l.txt' % (n_classifier, method))


def get_min(result):
    min_idx = -1
    min_val = 1
    for i in range(len(result)):
        if min_val > result[i]:
            min_val = result[i]
            min_idx = i

    print("min_idx:" + str(min_idx))
    print("min_val:" + str(min_val))
    return min_idx, min_val


def get_max(result):
    max_idx = -1
    max_val = 0
    for i in range(len(result)):
        if max_val < result[i]:
            max_val = result[i]
            max_idx = i

    print("max_idx:" + str(max_idx))
    print("max_val:" + str(max_val))
    return max_idx, max_val


idx = 10
print("%d %s" % (n_classifier, method))
min_idx, min_val = get_min(result[idx])
max_idx, max_val = get_max(result[idx])
