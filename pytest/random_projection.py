import _init_paths
from procedure.init_0.kmeans import base_multiple_kmeans, multiple_kmeans_batch
import numpy as np
import util4test
import matplotlib.pyplot as plt
import matplotlib as mat
import os
import time

n_cluster = 8
n_instance = 2
max_iter = 300
np.random.seed(1)


def random_projection():
    start_time = time.time()

    program_train_para_dir = '/home/bz/NN_as_Classification/pytest/data/kmeans_data'
    for i in range(n_instance):
        util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_1_' + str(i + 1))
    config = {
        "type": "kmeans",
        'specific_type': 'multiple_batch',
        "max_iter": max_iter,
        'n_cluster': n_cluster,
        'n_instance': n_instance,
        'dataset_partition': {
            'max_iter': max_iter
        },
        'entity_number': 1,
        'program_train_para_dir': program_train_para_dir,
        'kahip_dir': '/home/bz'
    }
    model = base_multiple_kmeans.BaseMultipleKMeans(config)
    centroid = np.random.normal(size=(n_cluster * n_instance, 2), loc=100, scale=100)
    print(centroid)
    idx = model.random_projection(centroid)
    print(idx)

    # 呈现未归集前的数据
    plt.scatter(centroid[:, 0], centroid[:, 1], s=50)
    plt.title("m=8, k=8 random projection")
    # plt.xticks(())
    # plt.yticks(())
    plt.savefig('data/before.jpg')

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
    plt.savefig('data/after.jpg')

    end_time = time.time()


def visualize_multiple_kmeans():
    os.system('rm -f figure/*')
    start_time = time.time()
    # base_dir = '/home/bz/NN_as_Classification/data/siftsmall_10/base.npy'
    # base = np.load(base_dir)
    base = np.random.normal(size=(1000, 2), loc=0, scale=100)

    program_train_para_dir = '/home/bz/NN_as_Classification/pytest/data/kmeans_data'

    util4test.delete_dir_if_exist(program_train_para_dir + '/Classifier_1_1')

    config = {
        "n_instance": n_instance,
        "n_cluster": n_cluster,
        "type": "kmeans",
        "specific_type": "multiple_batch",
        "dataset_partition": {
            "max_iter": max_iter
        },
        'entity_number': 1,
        'kahip_dir': '/home/bz/',
        'program_train_para_dir': program_train_para_dir
    }

    multiple_model = multiple_kmeans_batch.MultipleKMeansBatch(config)
    model_l = multiple_model.preprocess(base)
    # print(multiple_model.n_point_label)
    # print(multiple_model.centroids)
    # draw_cluster_point(multiple_model.centroids, get_color(multiple_model.centroids), 'multiple centroids')

    # partition = multiple_model.model.labels_
    # draw_cluster_point(base, partition, 'partition')

    for i, model in enumerate(model_l, 0):
        model.partition(base)
        print(model.n_point_label)
        # print(model.centroid_l)
        draw_cluster_point(base, model.labels, 'Classifier_' + str(i))
        draw_cluster_point(model.centroid_l, get_color(model.centroid_l), 'Classifier_' + str(i) + '_centroids')
        # model.save()

    end_time = time.time()
    print("time to train kmeans", (end_time - start_time))


def get_color(points):
    return ['red'] * len(points)


plt_idx = 1


# 根据聚类画图
def draw_cluster_point(base, labels, name):
    global plt_idx
    plt.figure(plt_idx)
    plt_idx += 1
    axes = plt.gca()
    axes.set_xlim([-430, 430])
    axes.set_ylim([-430, 430])

    plt.scatter(base[:, 0], base[:, 1], c=labels, s=10, cmap='Accent')
    # plt.xticks(())
    # plt.yticks(())
    plt.title("m=" + str(n_instance) + ", k=" + str(n_cluster) + " random projection")
    name = 'figure/' + name + '.jpg'
    plt.savefig(name)

    plt.close()


# random_projection()
visualize_multiple_kmeans()
