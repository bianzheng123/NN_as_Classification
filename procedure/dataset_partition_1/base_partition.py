import numpy as np


class BasePartition:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = config['save_dir']
        self.classifier_number = config['classifier_number']
        # key是每一个类的编号, value是属于该类的点在base对应的索引
        self.label = {}
        # 类的数量
        self.n_cluster = config['n_cluster']

    def partition(self, base):
        pass

    # 填充self.label, 就是根据cluster编号将base分成一类
    # 输入时需要转换成numpy格式
    def get_labels(self, labels):
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label[cluster_i] = base_idx_i

    def __str__(self):
        return '%s_%d, n_cluster: %d, save_dir: %s' % (self.type, self.classifier_number, self.n_cluster, self.save_dir)
