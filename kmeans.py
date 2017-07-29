import numpy as np
import matplotlib.pyplot as plt

#euclidean distance
def dist(point1,point2):
    return np.linalg.norm(point1-point2)

#kmeans, for color plot k < 9
def kmeans(array, k):
    initial = (np.random.randint(len(array),size=k))
    centroids = [array[i] for i in initial]
    cluster = []
    iteration = 0
    distortion = []
    
    #returns index j of closest cluster (distance from centroid)
    def closest(point):
        distances = [dist(point,centroids[j]) for j in range(k)]
        return distances.index(min(distances))
    
    #returns new centroid
    def cluster_mean(index):
        total = sum(array[i] for i in range(len(array)) if cluster[i]==index)
        return total/cluster.count(index)
        
    while True:
        iteration += 1
        cluster = [closest(array[i]) for i in range(len(array))]
        new_centroids = [cluster_mean(j) for j in range(k)]
        distortion.append(sum(dist(array[i], centroids[cluster[i]])**2 for i in range(len(array))))
        if np.array_equal(centroids,new_centroids):
            break
        else:
           centroids = new_centroids
    
    print ("initial point indices: ", initial)
    print ("final centroids: ", centroids)
    print ("number of iterations:", len(distortion))
    print ("change in distortion:", distortion)

    #plot
    colors = "bgrcmykw"
    for i in range(len(array)):
        plt.scatter(array[i][0],array[i][1], color = colors[cluster[i]])
    
    return distortion

def kmeans_ntimes(array, k, n):
    for i in range(n):
        print()
        print("Trial ", i+1)
        plt.plot(kmeans(array,k))


toyarray = np.loadtxt("toydata.txt")
kmeans(toyarray,3)    
#kmeans_ntimes(toyarray, 3, 10)
        



        