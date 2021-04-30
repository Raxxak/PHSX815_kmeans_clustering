import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.stats import poisson

s=[10,20,50,100,1000] # to create data sets of different sizes
for size1 in s:
    #create 2d data (poisson) centered at two parameters
    x = np.random.poisson(lam=(10., 30.), size=(size1, 2))


    #implement the clustering
    x.shape
    kmean=KMeans(n_clusters=2)

    kmean.fit(x)
    cc=kmean.cluster_centers_

    labels=kmean.labels_
    print("for "+str(size1) +"datapoints, the center is=",cc)
    
'''
# This code I found calculates the sum of squares within each cluster, did not have time to explore it 
wcss = []
for i in range(1,20):
 kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
 kmeans.fit(x)
 wcss.append(kmeans.inertia_)
 #print("Cluster", i, "Inertia", kmeans.inertia_)
 
plt.plot(range(1,20),wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') ##WCSS stands for total within-cluster sum of square
plt.show()

'''
