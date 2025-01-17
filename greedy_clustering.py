import numpy as np
### TODO: Add imports below
from multiprocessing import Pool, shared_memory
import time
import matplotlib.pyplot as plt
from memory_profiler import profile
### TODO: Add imports above

REP_FILE = 'representatives.csv'
ASGNMT_FILE = 'assignments.csv'

def euclidean(a,b):
    return np.linalg.norm(a-b)


### TODO: Add/Change functions below

def worker( start, end, centroids, tau, shared_mem_name, data_shape ):
    
    existing_shm = shared_memory.SharedMemory( name = shared_mem_name )
    
    data = np.ndarray( data_shape, buffer = existing_shm.buf )
    
    labels_chunk = []

    for i in range( start, end ):
        
        distances = [ euclidean( centroid, data[ i ] ) for centroid in centroids ]
        
        min_dist = min( distances )
        
        if min_dist > tau:
            labels_chunk.append( -1 )
        else:
            labels_chunk.append( np.argmin( distances ) )

    existing_shm.close()
    
    return labels_chunk


### TODO: Add/Change functions above

def greedy_clustering(file_path, tau, seed, nr_workers):
    """This fucntion applies the greedy clustering algorithm as inidcated in the Project instructions.

    Args:
        file_path (str): file path of the data set
        tau (int): threshold, if all distances between a data point and the already existing representatives is above the threshold, the data point becomes another representative.
        seed (int): seed for shuffeling
        nr_workers (int): number of workers/cores used for multiprocessing

    Returns:
        ratio (float): This is the ratio indicating how much smaller the reduced data set is. E.g. 0.1 means the reduced set has 10% of the capacity of the original data set. It is a value between 0 and 1.
    """

    ### Read Data
    print("Read data")
    data = np.loadtxt(file_path, delimiter=',')
    
    ## DeDuplication
    data = np.unique( data, axis=0 )

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(data)
    
    ### TODO: Insert code below
    assignments = []
    representatives = []
    
    shared_mem = shared_memory.SharedMemory( create = True, size = data.nbytes )
    shared_data = np.ndarray( data.shape, dtype = data.dtype, buffer = shared_mem.buf )
    
    #shared_data[:] = data[:]  
    np.copyto( shared_data, data )
    
    # Initialize centroids
    #centroids = [ data[0] ]
    centroids = [ rng.choice( data ) ]

    for point in data:
        if all( euclidean( point, c ) > tau for c in centroids ):
            centroids.append( point )

    chunk_size = len( data ) // nr_workers

    chunk_ranges = [ ( i * chunk_size, ( i + 1 ) * chunk_size if i < nr_workers - 1 else len( data ) )
                    for i in range( nr_workers ) ]

    with Pool( processes = nr_workers ) as pool:
        results_chunks = pool.starmap( worker, [ ( start, end, centroids, tau, shared_mem.name, data.shape )
                                       for start, end in chunk_ranges ] )

    assignments = np.concatenate( results_chunks )
    representatives = np.array( centroids )

    shared_mem.close()
    shared_mem.unlink()


    ### TODO: Insert code above
    
    ### Save preprocessed data
    ### NOTE: the object 'representatives' holds an iterable (list of lists or NumPy ndarray) with the selected representaives in shape (num_representives, 6).
    ### NOTE: the object 'data_with_labels' holds an iterable (list of lists or NumPy ndarray) with the data points and their assigned cluster (index of representative)
    ###  in shape (num_data_points, 7) where the last column is the index of the representative from the 'representative' object.
    data_with_labels = np.hstack(( data, np.array(assignments).reshape((-1,1))))
    
    np.savetxt(ASGNMT_FILE, data_with_labels, delimiter=",")
    np.savetxt(REP_FILE, representatives, delimiter=",")
     
    return len(representatives)/len(data)

if __name__ == "__main__":
    
    ### NOTE: The main clause will not be graded, change for your own convenience  
    ### TODO: Add/Change code below
    file_path = 'small_dataset.csv'
    workers = 4
    seed = 42
    tau = 2000
    #ratio = greedy_clustering(file_path, tau, seed, workers)
    #print(f"ratio: {ratio:.4f}")
    #assert (0.08 < greedy_clustering('dataset.csv', 200, 42, 4) < 0.09)
    #assert (0.08 < greedy_clustering('small_dataset.csv', 200, 42, 4) < 0.09) 


    data = np.loadtxt(file_path, delimiter=',')
    ratios = []
    runtimes = []
    
    taus = [0.01 * np.linalg.norm( np.ptp( data, axis=0 ) ),
            0.1 * np.linalg.norm( np.ptp( data, axis=0 ) ),
            0.25 * np.linalg.norm( np.ptp( data, axis=0 )  )]
    
    for tau in taus:
        #start_time = time.time()
        start_time = time.process_time()
        ratio = greedy_clustering( file_path, tau, seed, workers )
        # end_time = time.time()
        end_time = time.process_time()
        print(f"ratio: {ratio:.4f}")
        ratios.append( ratio )
        runtimes.append( end_time - start_time )

    plt.figure(figsize=(10, 6))
    plt.plot(runtimes, ratios, marker='o', linestyle='-')
   #plt.xlabel('Runtime (s)')
    #plt.ylabel('# of Representatives (|R|)')
    plt.title("# of Representatives vs Runtime for Different Tau Values", fontsize=14)
    plt.xlabel("Runtime (seconds)", fontsize=12)
    plt.ylabel("# of Representatives (|R| / |X|)", fontsize=12)

    #plt.title('# of Representatives vs Runtime for Different Tau Values')
    #plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(title="Tau Values", fontsize=10)
    plt.show()