from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Upload and clean our data
data = pd.read_csv('avocado.csv', index_col = 'Date')
data = data[data.columns[:-2]]
data['type'] = data['type'].replace({'conventional':0, 'organic':1})
print(data.head())

# Fit PCA and apply transformation
X = data.iloc[:,:-1].values
pca = PCA(n_components=2).fit(X)
X_t = pca.transform(X)

'''
# Plotting PCA transformed avocado dataset
plt.scatter(X_t[:,0],X_t[:,1])
plt.show()
'''

# We can also fit random projection!
rp = SparseRandomProjection() # n_components and epsilon are chosen for us
X2 = X = np.random.rand(100, 10000) # create very large random matrix (our avocado dataset only has 10 features so no need for RP)
X_rp = rp.fit_transform(X2)
print(X_rp.shape)

'''
# Plotting RP transformed random dataset
plt.scatter(X_rp[:,0],X_rp[:,1])
plt.show()
'''
