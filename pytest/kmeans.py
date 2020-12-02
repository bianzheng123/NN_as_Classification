import _init_paths
from procedure.dataset_partition_1.kmeans import multiple_kmeans, independent_kmeans
import numpy as np
from time import time
import os
import sklearn.cluster as cls
import util4test

n_cluster = 16
max_iter = 300


def train_kmeans():
    start_time = time()
    base_dir = '/home/bz/NN_as_Classification/data/siftsmall_10/base.npy'
    base = np.load(base_dir)

    program_train_para_dir = '/home/bz/NN_as_Classification/pytest/kmeans_data'

    util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_1_1')

    config = {
        "type": "kmeans",
        "max_iter": max_iter,
        'n_cluster': n_cluster,
        'n_instance': 1,
        'entity_number': 1,
        'program_train_para_dir': program_train_para_dir
    }

    model = independent_kmeans.IndependentKMeans(config)
    model.partition(base)
    model.save()
    for m in model.models:
        print(m.n_point_label)

    util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_2_1')

    config = {
        "type": "kmeans",
        "max_iter": max_iter,
        'n_cluster': n_cluster,
        'n_instance': 1,
        'entity_number': 2,
        'program_train_para_dir': program_train_para_dir
    }

    model = independent_kmeans.IndependentKMeans(config)
    model.partition(base)
    model.save()
    for m in model.models:
        print(m.n_point_label)

    end_time = time()
    print("time to train kmeans", (end_time - start_time))


def gnd_kmeans():
    start_time = time()
    base_dir = '/home/bz/NN_as_Classification/data/siftsmall_10/base.npy'
    base = np.load(base_dir)

    model = cls.KMeans(n_clusters=n_cluster, init='k-means++', max_iter=max_iter)
    model.fit(base)
    res = model.predict(base[0])
    print("centroid", model.cluster_centers_[:, 2])

    distance_table = [np.linalg.norm(base[0] - centroid) for centroid in self.centroid_l]
    distance_table = np.array([distance_table])

    # label = model.labels_
    # n_point_label = []
    # for cluster_i in range(16):
    #     base_idx_i = np.argwhere(label == cluster_i).reshape(-1)
    #     n_point_label.append(len(base_idx_i))
    # print(n_point_label)

    end_time = time()
    print("time to train kmeans", (end_time - start_time))


train_kmeans()
# gnd_kmeans()
