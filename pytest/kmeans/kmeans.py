import _init_paths
from procedure.init_0.kmeans import multiple_kmeans_batch
import numpy as np
from time import time
import os
import sklearn.cluster as cls
import util4test

n_cluster = 16
max_iter = 300


def train_kmeans():
    start_time = time()
    # base_dir = '/home/bz/NN_as_Classification/data/siftsmall_10/base.npy'
    # base = np.load(base_dir)
    base = np.random.normal(size=(1000, 2), loc=100, scale=100)

    program_train_para_dir = '/home/bz/NN_as_Classification/pytest/data/kmeans_data'

    util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_1_1')

    config = {
        "n_instance": 2,
        "n_cluster": 16,
        "type": "kmeans",
        "specific_type": "multiple_batch",
        "dataset_partition": {
            "max_iter": 40
        },
        'entity_number': 1,
        'kahip_dir': '/home/bz/',
        'program_train_para_dir': program_train_para_dir
    }

    multiple_model = multiple_kmeans_batch.MultipleKMeansBatch(config)
    model_l = multiple_model.preprocess(base)
    print(multiple_model.n_point_label)
    for model in model_l:
        model.partition(base)
        print(model.n_point_label)
        # model.save()

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
