from procedure.init_0.kmeans import base_multiple_kmeans
import numpy as np
from collections import defaultdict
import sklearn.cluster as cls


class MultipleKMeansSelfImpl(base_multiple_kmeans.BaseMultipleKMeans):

    def __init__(self, config):
        super(MultipleKMeansSelfImpl, self).__init__(config)
        # 用于构建m个kmeans的质心们, m * k * d
        self.model = KMeansPlusPlus(config)
        # self.centroid_l_l = None
        # self.model = None
        # self.signature = None


class KMeansPlusPlus:
    def __init__(self, config):
        self.n_cluster = config['n_cluster']
        self.max_iter = config['dataset_partition']['max_iter']
        self.tolerance_threshold = 1e-4
        self.n_init = 3  # 进行多次聚类，选择最好的一次
        self.random_state = np.random.mtrand._rand  # 随机数

        # 调用了fit之后才能获取这个
        self.cluster_centers_ = None

    def fit(self, base):
        self.tolerance_threshold = self._tolerance(base, self.tolerance_threshold)

        bestError = None
        bestCenters = None
        bestLabels = None
        for i in range(self.n_init):
            print("n_init", i)
            labels, centers, error = self._kmeans(base)
            if bestError == None or error < bestError:
                bestError = error
                bestCenters = centers
                bestLabels = labels
        self.cluster_centers_ = bestCenters

    # 把tol和dataset相关联
    def _tolerance(self, dataset, tol):
        variances = np.var(dataset, axis=0)
        return np.mean(variances) * tol

    # kmeans的主要方法，完成一次聚类的过程
    def _kmeans(self, base):
        self.base = np.array(base)
        bestError = None
        bestCenters = None
        bestLabels = None
        centerShiftTotal = 0
        centers = self._init_centroids(base)

        for i in range(self.max_iter):
            oldCenters = centers.copy()
            labels, error = self.update_labels_error(base, centers)
            centers = self.update_centers(base, labels)

            if bestError == None or error < bestError:
                bestLabels = labels.copy()
                bestCenters = centers.copy()
                bestError = error

            ## oldCenters和centers的偏移量
            centerShiftTotal = np.linalg.norm(oldCenters - centers) ** 2
            if centerShiftTotal <= self.tolerance_threshold:
                break

        # 由于上面的循环，最后一步更新了centers，所以如果和旧的centers不一样的话，再更新一次labels，error
        if centerShiftTotal > 0:
            bestLabels, bestError = self.update_labels_error(base, bestCenters)

        return bestLabels, bestCenters, bestError

    # kmeans++的初始化方式，加速聚类速度
    def _init_centroids(self, dataset):
        n_samples, n_features = dataset.shape
        centers = np.empty((self.n_cluster, n_features))
        # n_local_trials是每次选择候选点个数
        n_local_trials = None
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.n_cluster))

        # 第一个随机点
        center_id = self.random_state.randint(n_samples)
        centers[0] = dataset[center_id]

        # closest_dist_sq是每个样本，到所有中心点最近距离
        # 假设现在有3个中心点，closest_dist_sq = [min(样本1到3个中心距离),min(样本2到3个中心距离),...min(样本n到3个中心距离)]
        closest_dist_sq = distance(centers[0, np.newaxis], dataset)

        # current_pot所有最短距离的和
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.n_cluster):
            # 选出n_local_trials随机址，并映射到current_pot的长度
            rand_vals = self.random_state.random_sample(n_local_trials) * current_pot
            # np.cumsum([1,2,3,4]) = [1, 3, 6, 10]，就是累加当前索引前面的值
            # np.searchsorted搜索随机出的rand_vals落在np.cumsum(closest_dist_sq)中的位置。
            # candidate_ids候选节点的索引
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)

            # best_candidate最好的候选节点
            # best_pot最好的候选节点计算出的距离和
            # best_dist_sq最好的候选节点计算出的距离列表
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # 计算每个样本到候选节点的欧式距离
                distance_to_candidate = distance(dataset[candidate_ids[trial], np.newaxis], dataset)

                # 计算每个候选节点的距离序列new_dist_sq， 距离总和new_pot
                new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidate)
                new_pot = new_dist_sq.sum()

                # 选择最小的new_pot
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = dataset[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers

    # 更新每个点的标签，和计算误差
    def update_labels_error(self, dataset, centers):
        labels = self.assign_points(dataset, centers)
        new_means = defaultdict(list)
        error = 0
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            error += np.sqrt(np.sum(np.square(points - newCenter)))

        return labels, error

    # 更新中心点
    def update_centers(self, dataset, labels):
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            centers.append(newCenter)

        return np.array(centers)

    # 分配每个点到最近的center
    def assign_points(self, dataset, centers):
        labels = []
        for point in dataset:
            shortest = float("inf")  # 正无穷
            shortest_index = 0
            for i in range(len(centers)):
                val = distance(point[np.newaxis], centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            labels.append(shortest_index)
        return labels


def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2), axis=1))
