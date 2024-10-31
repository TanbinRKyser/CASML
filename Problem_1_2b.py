#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:22:38 2024

@author: tusker
"""

import multiprocessing
import random
from math import pi
import time
import matplotlib.pyplot as plt
#%%
def sample_pi(n):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of successes."""
    random.seed()  # Ensure each process has a unique seed
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s
#%%
def compute_pi(steps, num_worker):
    """ Return value should be the time measured for the execution """
    
    ### TODO: Add/change code below
    """ for i in range(os.cpu_count()):
            multiprocessing.Pool() """
    # List of core counts to test and lists to store results
    core_counts = [1, 2, 4, 8] # setting the sample cores
    measured_speedups = []
    theoretical_speedups = []
    serial_fraction = 0.1  # serial fraction for Amdahl's Law

    # Measure single-core (1 core) execution time for baseline
    steps_per_worker = steps  # All steps for one core
    
    random.seed( 1 ) 

    start_time = time.time()
    
    s_total = sample_pi( steps_per_worker )
    
    end_time = time.time()
    
    single_core_time = end_time - start_time

    # Iterate over core counts, measuring time for each
    for k in core_counts:
        if k == 1:
            measured_speedups.append(1)
            theoretical_speedups.append(1)
            continue

        # Parallel execution with k workers
        steps_per_worker = steps // k
        
        start_time = time.time()
        
        pool = multiprocessing.Pool( k ) # from the lecture slide
        results = pool.map( sample_pi, [steps_per_worker] * k ) # from the lecture slide

        end_time = time.time()
        
        # calculate total successes and time for this core count
        s_total = sum( results )
        
        parallel_time = end_time - start_time
        
        # estimate pi for the parallel execution
        pi_est = (4.0 * s_total) / steps
        print(f"Parallel Core Count: {k}")
        print(" Steps\tSuccess\tPi est.\tError")
        print("%6d\t%7d\t%1.5f\t%1.5f" % (steps, s_total, pi_est, pi - pi_est))
        
        measured_speedup = single_core_time / parallel_time
        measured_speedups.append( measured_speedup )
        
        # Amdahl Law
        theoretical_speedup = 1 / ( serial_fraction + ( 1 - serial_fraction ) / k)
        theoretical_speedups.append( theoretical_speedup )

    # Plot
    plt.figure( figsize=( 10, 6 ))
    plt.plot( core_counts, measured_speedups, 'b-', label="Measured Speedup" )
    plt.plot( core_counts, theoretical_speedups, 'r--', label="Theoretical Speedup" )
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup")
    plt.title("Monte Carlo pi estimation speedups")
    plt.xticks( core_counts )
    plt.legend()
    plt.grid()
    plt.show()

    # For the final output after measuring speedup
    if num_worker == 1 :
        measured_time = single_core_time 
    else : 
        measured_time = parallel_time

    ### TODO: Add/change code above

    return measured_time
#%%
if __name__ == "__main__":
    ### NOTE: The main clause will not be graded  
    ### TODO: Add/change test case to your convenience
    num_workers = 2
    #steps = 1000
    #steps = 100000
    steps = 1000000
    #steps = 10000000
    measured_time = compute_pi( steps, num_workers )
    print(f"Measured execution time with {num_workers} workers: {measured_time:.5f} seconds")




