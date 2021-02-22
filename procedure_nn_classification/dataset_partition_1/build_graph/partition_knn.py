import numpy as np
import faiss
from util import dir_io, read_data
import multiprocessing

np.random.seed(123)


class KNN:
    def __init__(self, config):
        self.k_graph = 40
        self.partition_depth = 1
        self.rp_miu = 0
        self.rp_sigma = 1
        # total gap is 10, this is the proportion
        self.random_gap = [3, 7]
        if 'k_graph' in config:
            self.k_graph = config['k_graph']
            print("k_graph %d" % self.k_graph)
        if 'partition_depth' in config:
            self.partition_depth = config['partition_depth']
            print("partition_depth %d" % self.partition_depth)
        if 'rp_miu' in config:
            self.rp_miu = config['rp_miu']
            print("rp_miu %d" % self.rp_miu)
        if 'rp_sigma' in config:
            self.rp_sigma = config['rp_sigma']
            print("rp_sigma %d" % self.rp_sigma)
        if 'random_gap' in config:
            self.random_gap = config['random_gap']
            print("random_gap %d %d" % (self.random_gap[0], self.random_gap[1]))

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, base_base_gnd, ins_intermediate):

        vertices = len(base)
        n_part = 2 ** self.partition_depth
        if vertices // n_part < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")
        if len(base) % (2 ** self.partition_depth) != 0:
            raise Exception("can not partition the dataset in this part")

        # use random projection to project the data to 2 part
        partition_idx = self.random_projection(base)

        self.partition_idx = partition_idx.reshape(n_part, -1)

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

    def random_projection(self, base):
        res_idx = np.arange(len(base))
        self.divide_and_conquer(0, base, 0, len(base), res_idx)
        return res_idx

    def divide_and_conquer(self, depth, data, start, end, res_idx):
        if depth == self.partition_depth:
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

    def graph_partition(self, config):
        # this function is to invoke kahip and read partition.txt
        save_dir = config['save_dir']
        graph_partition_type = config['graph_partition_type']
        n_cluster = config['n_cluster'] // (2 ** self.partition_depth)
        kahip_dir = config['kahip_dir']
        res_labels = np.empty([config['n_item']], dtype=np.int)

        for i in range(2 ** self.partition_depth):
            partition_dir = '%s/partition_%d.txt' % (save_dir, i)
            graph_dir = "%s/graph_%d.graph" % (save_dir, i)
            if graph_partition_type == 'kaffpa':
                kahip_command = '%s/deploy/kaffpa %s --preconfiguration=eco --output_filename=%s ' \
                                '--k=%d' % (
                                    kahip_dir, graph_dir,
                                    partition_dir,
                                    n_cluster)
                print(kahip_command)
                dir_io.kahip(partition_dir, kahip_command)
            elif graph_partition_type == 'parhip':
                kahip_command = 'mpirun -n %d %s/deploy/parhip %s --preconfiguration fastsocial ' \
                                '--save_partition --k %d' % (
                                    multiprocessing.cpu_count() // 2, kahip_dir, graph_dir,
                                    n_cluster)
                print(kahip_command)
                dir_io.kahip('./tmppartition.txtp', kahip_command)
                dir_io.move_file('tmppartition.txtp', partition_dir)
            tmp_labels = read_data.read_partition(partition_dir)
            for j in range(len(tmp_labels)):
                res_labels[self.partition_idx[i][j]] = tmp_labels[j] + n_cluster * i
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
