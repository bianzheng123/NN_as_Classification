import _init_paths
from procedure.dataset_partition_1.kmeans import base_multiple_kmeans
import numpy as np
import util4test
import matplotlib.pyplot as plt
import matplotlib as mat
import time

n_cluster = 8
n_instance = 8
max_iter = 300
np.random.seed(0)


def random_projection():
    start_time = time.time()

    program_train_para_dir = '/home/bz/NN_as_Classification/pytest/kmeans_data'
    for i in range(n_instance):
        util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_1_' + str(i + 1))
    config = {
        "type": "kmeans",
        "max_iter": max_iter,
        'n_cluster': n_cluster,
        'n_instance': n_instance,
        'entity_number': 1,
        'program_train_para_dir': program_train_para_dir
    }
    model = base_multiple_kmeans.BaseMultipleKMeans(config)
    centroid = np.random.normal(size=(n_cluster * n_instance, 2))
    print(centroid)
    idx = model.random_projection(centroid)
    print(idx)

    # 呈现未归集前的数据
    plt.scatter(centroid[:, 0], centroid[:, 1], s=50)
    plt.title("m=8, k=8 random projection")
    # plt.xticks(())
    # plt.yticks(())
    plt.savefig('before.jpg')

    centroid_after = centroid[idx]
    print(centroid_after)
    partition = np.empty(n_instance * n_cluster).astype(np.int32)
    for i in range(n_instance):
        for j in range(n_cluster):
            partition[i + j * n_instance] = i
    print(partition)
    plt.scatter(centroid_after[:, 0], centroid_after[:, 1], c=partition, s=50, cmap='Accent')
    # plt.xticks(())
    # plt.yticks(())
    plt.title("m=8, k=8 random projection")
    plt.savefig('after.jpg')

    end_time = time.time()


random_projection()
