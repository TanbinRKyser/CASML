#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:01:42 2024

@author: tusker
"""

import multiprocessing
import random
from math import pi
#import os
import time
#%%
def sample_pi(n, seed):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of successes."""
    ### TODO: Add/change code below
    random.seed( seed )  # Set the seed for reproducibility
    ### TODO: Add/change code above
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s
#%%
def compute_pi(steps, num_worker, seed):
    """ Return value should be the time measured for the execution """
    ### TODO: Add/change code below
    
    # Split the steps among the workers
    steps_per_worker = steps // num_worker
    
    # Start timing
    start_time = time.time()

    """ 
        pool = multiprocessing.Pool( num_worker )
        for i in range(num_worker):
            seeds = [seed + i]
    
        results = pool.map( sample_pi, [ ( steps_per_worker, s ) for s in seeds ] )
    """
    
    # Start the multiprocessing pool with the specified number of workers
    with multiprocessing.Pool( processes = num_worker ) as pool:
        # Create a list of seeds for each worker
        seeds = [ seed + i for i in range( num_worker ) ]
        
        # Run sample_pi in parallel with different seeds for each worker
        results = pool.starmap( sample_pi, [ ( steps_per_worker, s ) for s in seeds ] )
    
    # End timing
    end_time = time.time()
    
    measured_time = end_time - start_time
    
    # Sum the successful points from each worker
    s_total = sum( results )
    
    # Estimate Pi using the total successes
    pi_est = ( 4.0 * s_total ) / steps
    print(" Steps\tSuccess\tPi est.\tError")
    print("%6d\t%7d\t%1.5f\t%1.5f" % ( steps, s_total, pi_est, pi - pi_est))

    ### TODO: Add/change code above
    return measured_time  # Return pi_est instead of time for simplicity in testing
#%%
if __name__ == "__main__":
    ### NOTE: The main clause will not be graded  
    ### TODO: Add/change test case to your convenience
    steps = 1000 #0.03s
    #steps=10000000 #1.26318s
    #steps=100000000 #19.96878s

    #num_workers = 2
    num_workers = 4
    
    seed = 1
    
    measured_time = compute_pi(steps, num_workers, seed)
    print(f"Execution time with {num_workers} workers: {measured_time:.5f} seconds")