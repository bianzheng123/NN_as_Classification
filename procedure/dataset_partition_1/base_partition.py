import numpy as np
import os


class BasePartition:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = '%s/dataset_partition' % config['save_dir']
        os.system('mkdir %s' % self.save_dir)
        self.classifier_number = config['classifier_number']
        # key是每一个类的编号, value是属于该类的点在base对应的索引
        self.label_map = {}
        # 类的数量
        self.n_cluster = config['n_cluster']
        self.labels = None

    def partition(self, base):
        pass

    # 填充self.label, 就是根据cluster编号将base分成一类
    # 输入时需要转换成numpy格式
    def get_labels(self, labels):
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i

    def save(self):
        save_label_dir = '%s/partition.txt' % self.save_dir
        np.savetxt(save_label_dir, self.labels, fmt='%i')

    def __str__(self):
        return '%s_%d, n_cluster: %d, save_dir: %s' % (self.type, self.classifier_number, self.n_cluster, self.save_dir)
