import numpy as np
import os
import time


class BasePartition:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = '%s/dataset_partition' % config['save_dir']
        os.system('mkdir %s' % self.save_dir)
        self.classifier_number = config['classifier_number']
        self.entity_number = config['entity_number']
        self.obj_id = "%s_%d_%d" % (self.type, self.entity_number, self.classifier_number)
        # 类的数量
        self.n_cluster = config['n_cluster']
        self.model_info = None
        # key是每一个类的编号, value是属于该类的点在base对应的索引
        self.label_map = {}
        # 计数不同桶中点的数量
        self.n_point_label = None

        self.labels = None

    def partition(self, base):
        start_time = time.time()
        print('start dataset partitioning %s' % self.obj_id)
        self._partition(base)
        self.get_labels(self.labels)
        print('finish dataset partitioning %s' % self.obj_id)
        end_time = time.time()
        intermediate_config = {
            'time': end_time - start_time,
            'cluster_number_distribution': self.n_point_label
        }
        model_info = {
            "classifier_number": self.classifier_number,
            "entity_number": self.entity_number,
        }
        return (self.labels, self.label_map), (model_info, intermediate_config)

    '''
    生成self.labels, 就是对每一个item赋值label
    '''

    def _partition(self, base):
        pass

    # 填充self.label, 就是根据cluster编号将base分成一类
    # 输入时需要转换成numpy格式
    def get_labels(self, labels):
        self.n_point_label = []
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))

    def save(self):
        save_label_dir = '%s/partition.txt' % self.save_dir
        np.savetxt(save_label_dir, self.labels, fmt='%i')
        # save_distribution_dir = '%s/distribution_partition.txt' % self.save_dir
        # np.savetxt(save_distribution_dir, self.n_point_label, fmt='%i')

    def __str__(self):
        return '%s, n_cluster: %d, save_dir: %s' % (self.obj_id, self.n_cluster, self.save_dir)
