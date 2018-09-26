from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import helper

data = pd.read_csv('bmi_and_life_expectancy.csv', index_col='Country')

def plot_clusters(data, n):
    X = data.values

    # TODO: Create an instance of KMeans to find two clusters
    kmeans_1 = KMeans(n_clusters=n).fit(X)

    # TODO: use fit_predict to cluster the dataset
    predictions = kmeans_1.predict(X)

    # Plot
    #draw_clusters(data.iloc[:,:2], predictions)
    clustered = pd.concat([data.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    plt.scatter(X[:,0], X[:,1], c=clustered['group'], s=20, cmap='viridis')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.show()

# Play around with different params to understand data
plot_clusters(data,10)
