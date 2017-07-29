import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

#gaussian, for color plot k < 9
def gaussian(array, k):
    
    #initialization
    pij = np.random.dirichlet(np.array([1]*k), size = len(array))
    print ("initial pij: ", pij)
    
    priors = np.mean(pij, axis = 0)
    means = []
    covariances = []
    for j in range(k):
        means.append(sum(array[i]*pij[i][j] for i in range(len(array)))/np.sum(pij,axis=0)[j])    
        covariances.append(sum(np.outer(array[i]-means[j],array[i]-means[j])*pij[i][j] for i in range(len(array)))/np.sum(pij,axis=0)[j])
    cluster = np.argmax(pij, axis=1)
    iteration = 0
    new_pij = np.zeros((len(array),k))

    #EM algo
    while True:
        iteration += 1
        
        #E-step, update the p_i,j
        for i in range(len(array)):
            denominator1 = sum(multivariate_normal.pdf(array[i],means[j],covariances[j])*priors[j] for j in range(k))
            for j in range(k):
                new_pij[i][j] = multivariate_normal.pdf(array[i],means[j],covariances[j])*priors[j]/denominator1
        cluster = np.argmax(new_pij, axis=1)
        
        #M-step, update priors, means, convariances
        new_priors = np.mean(new_pij, axis = 0)
        for j in range(k):
            denominator = np.sum(new_pij,axis=0)[j]
            means[j] = sum(array[i]*new_pij[i][j] for i in range(len(array)))/denominator
            covariances[j] = sum(np.outer(array[i]-means[j],array[i]-means[j])*new_pij[i][j] for i in range(len(array)))/denominator
        if np.array_equal(priors, new_priors):
            break
        else:
            priors = new_priors
    
    print ("iterations: ", iteration)
    print ("final: ", new_pij)
    print ("final clusters: ", cluster)
    
    #plot
    colors = "bgrcmykw"
    for i in range(len(array)):
        plt.scatter(array[i][0],array[i][1], color = colors[cluster[i]])
    

toyarray = np.loadtxt("toydata.txt")
gaussian(toyarray,3)    
        



        