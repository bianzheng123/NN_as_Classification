import numpy as np
import faiss
from util import dir_io, read_data
import multiprocessing
import sklearn.cluster as cls

np.random.seed(123)


class KNN:
    def __init__(self, config):
        self.k_graph = 40
        self.partition_iter = 1
        if 'k_graph' in config:
            self.k_graph = config['k_graph']
            print("k_graph %d" % self.k_graph)
        if 'partition_iter' in config:
            self.partition_iter = config['partition_iter']
            print("partition_iter %d" % self.partition_iter)

        self.partition_idx = None

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, base_base_gnd, ins_intermediate):

        vertices = len(base)
        n_part = 2 ** self.partition_iter
        if vertices // n_part < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")
        if len(base) % (2 ** self.partition_iter) != 0:
            raise Exception("can not partition the dataset in this part")

        # partition the data to (self.partition_iter**2) part
        self.partition_idx = self.partition_dataset(base)

        # build sub graph and partition the graph
        res_graph_l = []
        for i in range(n_part):
            base_tmp = np.array(base[self.partition_idx[i]])

            dim = base_tmp.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(base_tmp)
            distance, index_arr = index.search(base_tmp, self.k_graph + 1)  # first index should be it self, so +1
            index_arr = index_arr[:, :] + 1  # kahip require the order start from 1, so add 1 in total
            weightless_graph = index_arr.tolist()
            for i in range(len(weightless_graph)):
                weightless_graph[i] = set(weightless_graph[i])

            # print("get the nearest k result")

            for i in range(len(weightless_graph)):
                if (i + 1) in weightless_graph[i]:
                    weightless_graph[i].remove((i + 1))
                for vertices_index in weightless_graph[i]:
                    if (i + 1) not in weightless_graph[vertices_index - 1]:
                        weightless_graph[vertices_index - 1].add(i + 1)

            res_graph = []
            for i in range(len(weightless_graph)):
                tmp_line = {}
                for vertices in weightless_graph[i]:
                    tmp_line[vertices] = 1
                res_graph.append(tmp_line)

            res_graph_l.append(res_graph)
            # print("change the rank into graph successfully")
        return res_graph_l

    def partition_dataset(self, base):
        pass

    def count_proportion(self, partition_idx_l, n_part, n_cluster):
        partition_len_l = np.array([len(_) for _ in partition_idx_l])
        len_sum = np.sum(partition_len_l)
        partition_proportion_l = 1.0 * partition_len_l / len_sum
        res_proportion = np.round(partition_proportion_l * (n_cluster - n_part)).astype(np.int) + 1
        assert np.sum(res_proportion) == n_cluster
        return res_proportion

    def graph_partition(self, config):
        # this function is to invoke kahip and read partition.txt
        save_dir = config['save_dir']
        graph_partition_type = config['graph_partition_type']
        n_part = 2 ** self.partition_iter
        n_cluster_l = self.count_proportion(self.partition_idx, n_part, config['n_cluster'])
        print(n_cluster_l)
        kahip_dir = config['kahip_dir']
        res_labels = np.empty([config['n_item']], dtype=np.int)
        # for efficiently count the offset of different labels
        n_cluster_cumsum_l = np.insert(n_cluster_l, 0, values=0, axis=0)
        n_cluster_cumsum_l = np.cumsum(n_cluster_cumsum_l)

        for i in range(n_part):
            partition_dir = '%s/partition_%d.txt' % (save_dir, i)
            graph_dir = "%s/graph_%d.graph" % (save_dir, i)
            if graph_partition_type == 'kaffpa':
                kahip_command = '%s/deploy/kaffpa %s --preconfiguration=eco --output_filename=%s ' \
                                '--k=%d' % (
                                    kahip_dir, graph_dir,
                                    partition_dir,
                                    n_cluster_l[i])
                print(kahip_command)
                dir_io.kahip(partition_dir, kahip_command)
            elif graph_partition_type == 'parhip':
                kahip_command = 'mpirun -n %d %s/deploy/parhip %s --preconfiguration fastsocial ' \
                                '--save_partition --k %d' % (
                                    multiprocessing.cpu_count() // 2, kahip_dir, graph_dir,
                                    n_cluster_l[i])
                print(kahip_command)
                dir_io.kahip('./tmppartition.txtp', kahip_command)
                dir_io.move_file('tmppartition.txtp', partition_dir)
            tmp_labels = read_data.read_partition(partition_dir)
            for j in range(len(tmp_labels)):
                res_labels[self.partition_idx[i][j]] = tmp_labels[j] + n_cluster_cumsum_l[i]
        partition_dir = '%s/partition.txt' % save_dir
        dir_io.save_array_txt(partition_dir, res_labels, '%d')
        return res_labels

    @staticmethod
    def save(graph_l, save_dir):
        for i in range(len(graph_l)):
            graph = graph_l[i]
            # graph is the 2d array
            vertices = len(graph)
            edges = 0
            for vecs in graph:
                edges += len(vecs)
            assert edges % 2 == 0
            edges = edges / 2

            tmp_save_dir = '%s/graph_%d.graph' % (save_dir, i)
            dir_io.save_graph(tmp_save_dir, graph, vertices, edges)
            print("graph building complete")


class KNNRandomProjection(KNN):
    def __init__(self, config):
        super(KNNRandomProjection, self).__init__(config)
        self.rp_miu = 0
        self.rp_sigma = 1
        # total gap is 10, this is the proportion
        self.random_gap = [3, 7]
        if 'rp_miu' in config:
            self.rp_miu = config['rp_miu']
            print("rp_miu %d" % self.rp_miu)
        if 'rp_sigma' in config:
            self.rp_sigma = config['rp_sigma']
            print("rp_sigma %d" % self.rp_sigma)
        if 'random_gap' in config:
            self.random_gap = config['random_gap']
            print("random_gap %d %d" % (self.random_gap[0], self.random_gap[1]))

    def partition_dataset(self, base):
        res_idx = np.arange(len(base))
        self.divide_and_conquer(0, base, 0, len(base), res_idx)
        n_part = 2 ** self.partition_iter
        res_idx = res_idx.reshape(n_part, -1)
        return res_idx

    def divide_and_conquer(self, depth, data, start, end, res_idx):
        if depth == self.partition_iter:
            return
        # vector = np.random.rand(data.shape[1])
        # random_vector = vector / np.linalg.norm(vector)
        random_vector = np.random.normal(size=data.shape[1], scale=self.rp_sigma, loc=self.rp_miu)
        random_l = []
        for i in range(start, end):
            random_num = np.dot(random_vector, data[res_idx[i]])
            random_l.append(random_num)
        # random_l is the result of dot product of centroid and random vector(follow Gauss distribution)
        random_l = np.array(random_l)
        depth += 1
        sort_indices = np.argsort(random_l) + start

        random_start = int((end - start) / 10 * self.random_gap[0] + start)
        random_end = int((end - start) / 10 * self.random_gap[1] + start)
        mid = np.random.randint(random_start, random_end)
        res_idx[start:end] = sort_indices
        self.divide_and_conquer(depth, data, start, mid, res_idx)
        self.divide_and_conquer(depth, data, mid, end, res_idx)


class KNNLSH(KNN):
    def __init__(self, config):
        super(KNNLSH, self).__init__(config)
        self.r = 1
        self.a_sigma = 1
        self.a_miu = 0
        if 'r' in config:
            self.r = config['r']
            print("r %d" % self.r)
        if 'a_sigma' in config:
            self.a_sigma = config['a_sigma']
            print("a_sigma %d" % self.a_sigma)
        if 'a_miu' in config:
            self.a_miu = config['a_miu']
            print("a_miu %d" % self.a_miu)

    def partition_dataset(self, base):
        res_idx = self.lsh(base, 2 ** self.partition_iter)
        return res_idx

    def lsh(self, base, n_part):
        norm = np.linalg.norm(base, axis=1)
        norm_div = np.max(norm)
        base_normlize = base / norm_div
        self.a = np.random.normal(loc=self.a_miu, scale=self.a_sigma, size=base.shape[1])
        proj_result = np.dot(base_normlize, self.a)
        self.b = np.random.random() * self.r
        label = np.floor((proj_result + self.b) / self.r) % n_part

        res_idx = []
        for i in range(n_part):
            base_idx_i = np.argwhere(label == i).reshape(-1)
            res_idx.append(base_idx_i)
        return res_idx


class KNNKMeans(KNN):
    def __init__(self, config):
        super(KNNKMeans, self).__init__(config)
        self.max_iter = 40
        if 'max_iter' in config:
            self.max_iter = config['max_iter']
            print("max_iter %d" % self.max_iter)

    def partition_dataset(self, base):
        res_idx = self.kmeans(base, 2 ** self.partition_iter)
        return res_idx

    def kmeans(self, base, n_part):
        model = cls.KMeans(n_clusters=n_part, init='k-means++', max_iter=self.max_iter)
        model.fit(base)
        label = model.labels_

        res_idx = []
        for i in range(n_part):
            base_idx_i = np.argwhere(label == i).reshape(-1)
            res_idx.append(base_idx_i)
        return res_idx
