import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
from sklearn.cluster import KMeans
import numpy as np

# 1. 产生模拟数据
k = 5
X, Y = make_blobs(n_samples=1000, n_features=2, centers=k, random_state=1)
# print(X)
# print(Y)

# 2. 模型构建
km = KMeans(n_clusters=k, init='k-means++', max_iter=30)
km.fit(X)

# 获取簇心
centroids = km.cluster_centers_
# print(centroids)
para = km.get_params()
print(para)
# print(km.score(X))
# 获取归集后的样本所属簇对应值

# labels = km.labels_
# print(labels.shape)

query = np.array([[1, 2], [3, 4]])
# print(query)
y_kmean = km.predict(query)
# print(y_kmean)
query = km.transform(query)
# print(query)
# print(np.argsort(query))

# 呈现未归集前的数据
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.yticks(())
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5)
plt.show()
