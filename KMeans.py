from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

data = pd.read_csv('bmi_and_life_expectancy.csv', index_col='Country')

def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg

def plot_clusters(data, n):
    X = data.values

    # TODO: Create an instance of KMeans to find two clusters
    kmeans_1 = KMeans(n_clusters=n).fit(X)

    # TODO: use fit_predict to cluster the dataset
    predictions = kmeans_1.predict(X)

    # Plot
    clustered = pd.concat([data.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    plt.scatter(X[:,0], X[:,1], c=clustered['group'], s=20, cmap='viridis')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.show()


'''
# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, 30, 2)

# Calculate error values for all k values we're interested in
X = data.values
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]

# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlabel('K - number of clusters')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.plot(possible_k_values, errors_per_k)

# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')
'''
# Play around with different params to understand data
plot_clusters(data,6)
