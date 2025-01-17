from greedy_clustering import greedy_clustering, euclidean
import numpy as np
### TODO: Add imports below
from multiprocessing import Pool, shared_memory
import time
import matplotlib.pyplot as plt
### TODO: Add imports above

REP_FILE = 'representatives.csv'
ASGNMT_FILE = 'assignments.csv'
K_MEANS_CENTROID_FILE = 'kmeans_centroids.csv'
K_MEANS_LABELS_FILE = 'kmeans_labels.csv'
FINAL_ASSGMNT_GREEDY_FILE = 'final_assignment_a1.csv'
FINAL_ASSGMNT_SUBSAMPLING_FILE = 'final_assignment_a2.csv'

def subsampling(file_path, ratio, seed):
    print("Read data")
    data = np.loadtxt(file_path, delimiter=',')
    rng = np.random.default_rng(seed=seed)

    assignments = None
    ### TODO: Insert code below
    representatives = []

    #Hier, wir haben ratio = 0.01, also 1% der Datenpunkte
    num_samples = int( ratio * len( data ) )
    
    representatives = rng.choice( data, size = num_samples, replace = False )

    ### TODO: Insert code above
    np.savetxt(REP_FILE, representatives, delimiter=",")

    #return representatives
    return None


### TODO: Add/Change functions below
# def worker( start, end, centroids, tau, shared_mem_name, data_shape ):
# def worker( data_chunk, centroids, variations ):
def worker( data_chunk, centroids ):
    #pass
    labels = []

    for p in  data_chunk:
        dist = [ euclidean( centroid, p ) for centroid in centroids ]
        labels.append( np.argmin( dist ) )

    return labels

### TODO: Add/Change functions above



def k_means(R_file, K, max_iters, seed):
    R = np.loadtxt(R_file, delimiter=",")

    centroids = np.zeros((K, 6), dtype=float)
    labels = np.zeros(len(R), dtype=int)
    rng = np.random.default_rng(seed=seed)

    ### TODO: Insert code below
    
    # centroids = rng.choice(R, size=K, replace=False)
    # for i in range( max_iters ):
    #     labels = np.array( worker( data, centroids ) )
    #     for k in range( K ):
    #         cp = data[ labels == k ]
    #         if len( cp ) > 0:
    #             centroids[ k ] = cp.mean( axis=0 )
    
    # wenn centroids nicht zufällig initialisiert werden, wird die Konvergenz zunächst fehlschlagen
    centroids = R[ rng.choice( len( R ), K, replace = False ) ]
    
    
    nr_workers = 4
    chunk_size = len( R ) // nr_workers
    
    
    # chunks = [ R[ i : i + chunk_size] for i in range(0, len( R ), chunk_size )]

    # for iteration in range( max_iters  ):
    #     with Pool( processes = nr_workers ) as pool:
    #         # Parallel label assignment using starmap
    #         results = pool.starmap( worker, [ (chunk, centroids) for chunk in chunks ] )
    #         labels = np.concatenate(results)
 
    #     # Update centroids based on assigned clusters
    #     for k in range( K ):
    #         cp = R[  labels == k ]
    #         if len( cp ) > 0:
    #             centroids[k] = cp.mean(axis=0)
    
    for _ in range( max_iters ):
        with Pool( processes = nr_workers ) as pool:         
            chunks = [ R[ i * chunk_size: ( i + 1 ) * chunk_size ] for i in range( nr_workers ) ]

            if len( R ) % nr_workers != 0:  
                chunks[-1] = np.vstack( ( chunks[ -1 ], R[ nr_workers * chunk_size: ] ) )

            assignments = pool.starmap( worker, [ ( chunk, centroids ) for chunk in chunks ] )

            labels = np.concatenate( assignments )

        for k in range(K):
            #cp = R[ labels == k ]
            cp = R[ np.where( labels == k )[ 0 ] ]

            if len( cp ) > 0:
                centroids[ k ] = cp.mean( axis = 0 )
   
    ### TODO: Insert code above
    ### NOTE: the object 'centroids' holds an iterable (list of lists or NumPy ndarray) with the selected centroids in shape (num_centroids, 6).
    ### NOTE: the object 'labels' holds an iterable (list/array) with the cluster labels for each representative
    ### NOTE: This ONLY works when the order of representatives is preserved.
    np.savetxt(K_MEANS_CENTROID_FILE, centroids, delimiter=",")
    np.savetxt(K_MEANS_LABELS_FILE, labels, delimiter=",")

    return None


def remap_labels(kmeans_labels_file, kmeans_centroids_file, assignments_file, dataset_path, subsample):
    if not subsample:
        ### NOTE: assignments shape is (num_data_points, 7)
        assignments = np.loadtxt(assignments_file, delimiter=",")
    else:
        ### NOTE: assignments shape is (num_data_points, 6)
        ### NOTE: since the subsampling doesn't save the assignments, it is only the datapoints.
        assignments = np.loadtxt(dataset_path, delimiter=",")
    
    kmeans_centroids = np.loadtxt(kmeans_centroids_file, delimiter=",")
    
    ### NOTE: kmeans_labels has length num_representatives
    kmeans_labels = np.loadtxt(kmeans_labels_file, delimiter=",")

    ### NOTE: X_labels has length num_data_points
    X_labels = np.zeros(len(assignments), dtype=int)
    ### NOTE: final_assignments has shape (num_data_points,7) , 6 dimensions for data points and 1 dimension for cluster label of kmeans
    final_assignments = np.zeros((len(assignments), 7))
    
    ### TODO: Insert code below

    # if kmeans_centroids.shape[0] != len(kmeans_labels):
    #     raise ValueError("Mismatch between centroids and labels length!")

    for i, point in enumerate( assignments ): 
        #point_features = point[ : -1 ] if point.shape[ 0 ] == 7 else point
        #point_features = point[:-1] if not subsample else point
        point_features = point if subsample else point[:-1]
        distances = [ euclidean( point_features, centroid ) for centroid in kmeans_centroids ]
        #X_labels = np.argmin( distances )
        X_labels = point_features
        final_assignments[i, :-1] = point_features
        final_assignments[i, -1] = np.argmin( distances )
    
   
    ### TODO: Insert code above
    if subsample:
        np.savetxt(FINAL_ASSGMNT_SUBSAMPLING_FILE, final_assignments, delimiter=",")
    else:
        np.savetxt(FINAL_ASSGMNT_GREEDY_FILE, final_assignments, delimiter=",")

    return None

if __name__ == "__main__":
    
    ### NOTE: The main clause will not be graded, change for your own convenience  
    ### TODO: Add/Change code below

    file_path = 'small_dataset.csv' 
    subsample = True

    nr_workers = 4

    # preprocessing
    if subsample:
        subsampling(file_path, ratio=0.01, seed=1)
    else:
        #greedy_clustering(file_path, tau=100, dist_func=euclidean, seed=1)
        greedy_clustering( file_path, tau = 100, seed = 1, nr_workers = nr_workers  )

    #start= time.time()
    start = time.process_time()
    # k-means on R
    k_means(REP_FILE, K=3, max_iters=100, seed=1)
    #end = time.time()
    end = time.process_time()
    print("Time taken for k-means: ", end-start)

    # postprocessing
    X_labels = remap_labels(K_MEANS_LABELS_FILE, K_MEANS_CENTROID_FILE, ASGNMT_FILE, file_path, subsample=False)
