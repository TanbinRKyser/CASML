import numpy as np


def expected_samples(n):
    '''Returns theoretical value for number of samples '''
    
    ### TODO: Add/Change code below
    #num_samples = None # count of samples taken until the set is full

    num_samples = sum( 1 / i for i in range( 1, n + 1 ) ) * n ## ratio: 1.0010

    # EM constant
    # gamma = 0.5772156649 
    # harmonic_nummer = np.log( n ) + gamma
    # num_samples = harmonic_nummer * n 
    # 
    # ## ratio: 1.0009
    ### TODO: Add/Change code above
    
    return np.ceil(num_samples)



def experimental_samples(n):
    '''Runs the experiment until the set 1,...,n is full. Returns the number of samples needed'''

    ### TODO: Add/Change code below
    #num_samples = None # count of samples taken until the set is full
    unique_items = set() 
    num_samples = 0

    while len( unique_items ) < n:
        #num_samples += 1
        unique_items.add( np.random.randint( 1, n + 1 ) )
        num_samples += 1
    ### TODO: Add/Change code above
    
    return num_samples

def run_simulations(num_simulations, n, seed):
    
    np.random.seed(seed)
    
    ### TODO: Add/Change code below
    #avg_samples = None
    #total_samples = sum( experimental_samples( n   ) for _ in range( num_simulations )  )
    total_samples = 0
    
    for _ in range( num_simulations ):
        total_samples += experimental_samples( n )
    
    #print(f'tot_samples {total_samples}')
    
    avg_samples = total_samples / num_simulations
    ### TODO: Add/Change code above
    
    return np.ceil(avg_samples)


if __name__ == "__main__":
    ### NOTE: The main clause will not be graded, change for your own convenience  
    ### TODO: Add/Change code below

    n = 1000
    num_simulations = 100
    
    expected_value = expected_samples( n )
    simulated_value = run_simulations( num_simulations, n, 42 )
    
    print(expected_value/simulated_value)
    #print(f"exp_val    : {ev}")
    #print(f"sum_val: {sv}")
    #print(f"ratio : {ev / sv}")