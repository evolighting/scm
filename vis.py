import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

uniform_data = (index_cluster.index_expression * np.log2(10))[si, :]
ax = sns.heatmap(uniform_data, yticklabels=False, cmap='bwr')
plt.show()