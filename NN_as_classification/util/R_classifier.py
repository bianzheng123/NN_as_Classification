import json
import matplotlib.pyplot as plt
import numpy as np


def get_cluster_other(json_dir):
    x_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['recall'] == 0.0:
                continue
            x_arr.append(ele['n_candidates_avg'])
            y_arr.append(ele['recall'])
    return x_arr, y_arr


def get_cluster_nn_classification(json_dir):
    x_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['recall'] == 0.0:
                continue
            x_arr.append(ele['n_candidate'])
            y_arr.append(ele['recall'])
    return x_arr, y_arr


other_fname = '../result/other/graph_kmeans_kmm/'
dir_learn_on_graph = other_fname + 'learn-on-graph_0.json'
nn_classification_fname = '../result/8_kmeans_16_cluster/'
dir_kmeans = nn_classification_fname + 'result.json'

cls_learn_on_graph = get_cluster_other(dir_learn_on_graph)
cls_kmeans = get_cluster_nn_classification(dir_kmeans)

# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
line1_1, = plt.plot(cls_learn_on_graph[0], cls_learn_on_graph[1], marker='o', linestyle='solid', color='#b9529f',
                    label='16 bin space partition for NN')

line2_1, = plt.plot(cls_kmeans[0], cls_kmeans[1], marker='^', linestyle='solid', color='#3953a4',
                    label='8 kmeans 16 cluster self-implementation')

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
plt.title('graph kmeans vs knn')
plt.legend(loc='lower right', title="SIFT10K, 10-NN")

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.show()
