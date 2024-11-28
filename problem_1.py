#!/usr/bin/env python
#
# File: problem_1.py
# Author: Alexander Schliep (alexander@schlieplab.org)
#
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing

def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples = n, centers = c, cluster_std=1.7, shuffle=False,
                    random_state = 2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)

### TODO: Add/Change function below
def worker(data_chunk, centroids):
    """Assign data points to nearest centroids for a chunk of data."""
    cluster_assignments = []
    distances = []
    for datum in data_chunk:
        cluster, dist = nearestCentroid(datum, centroids)
        cluster_assignments.append(cluster)
        distances.append(dist**2)
    return cluster_assignments, distances
### TODO: Add/Change function above

def kmeans(k, data, nr_iter = 100, workers=None):
    """K-means clustering

    Args:
        k (int): number of clusters to form
        data (np.ndarray): Data points in shape (samples, 2)
        nr_iter (int, optional): _description_. Defaults to 100.
        workers (int): number of parallel processes

    Returns:
        total variation: sum of all squared distances
        c: cluster assignments, array of length N (number of data points)
    """
    ### TODO: Add/Change code below

    N = len(data)

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)),size=k,replace=False)]
    logging.debug("Initial centroids\n", centroids)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0
    for j in range(nr_iter):
        logging.debug("=== Iteration %d ===" % (j+1))

        # Parallel assignment step
        chunk_size = N // workers if workers else N
        data_chunks = [data[i:i + chunk_size] for i in range(0, N, chunk_size)]

        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.starmap(worker, [(chunk, centroids) for chunk in data_chunks])
        

        """ # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        cluster_assignments = {x: [] for x in range(k)}
        for i in range(N):
            cluster, dist = nearestCentroid(data[i],centroids)
            c[i] = cluster
            cluster_assignments[cluster].append(data[i])
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2
        """
        cluster_assignments = []
        distances = []
        for clusters, dists in results:
            cluster_assignments.extend(clusters)
            distances.extend(dists)
        
        c = np.array(cluster_assignments)
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)

        for cluster, dist in zip(c, distances):
            variation[cluster] += dist
            cluster_sizes[cluster] += 1
        
        delta_variation = -total_variation
        total_variation = sum(variation) 
        delta_variation += total_variation
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        centroids = np.zeros((k,2)) # This fixes the dimension to 2
        for i in range(N):
            centroids[c[i]] += data[i]        
        centroids = centroids / cluster_sizes.reshape(-1,1)
        
        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)
    
    ### TODO: Add/Change function above
    return total_variation, c


def computeClustering(samples, k_clusters, iterations, classes, workers, plot, verbose, debug):
    if verbose:
        logging.basicConfig(format='# %(message)s',level=logging.INFO)
    if debug: 
        logging.basicConfig(format='# %(message)s',level=logging.DEBUG)

    
    X = generateData(samples, classes)

    runtimes = []
    workers_list = list(range(1, workers + 1))

    for worker in workers_list:
        start_time = time.time()
        total_variation, assignment = kmeans(k_clusters, X, nr_iter = iterations, workers = worker)
        end_time = time.time()
        runtime = end_time-start_time
        runtimes.append(runtime)

    # Calculate speedup
    single_thread_time = runtimes[0]
    speedup = [single_thread_time / runtime for runtime in runtimes]
    #logging.info(single_thread_time)

    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print(f"Total variation {total_variation}")

    plt.figure()
    plt.plot(workers_list, speedup, marker='o')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title('K-Means Speedup with Parallelization')
    plt.grid(True)
    #if plot:
        #plt.savefig(plot)
    plt.show()

    if plot: # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title("k-means result")
        #plt.show()        
        fig.savefig(plot)
        plt.close(fig)

if __name__ == "__main__":
    
    workers = 4
    samples = 50000
    k_clusters = 3
    iterations = 100
    classes = 3
    plot = "test.png"
    verbose = True
    debug = False
    computeClustering(samples, k_clusters, iterations, classes, workers, plot, verbose, debug)

