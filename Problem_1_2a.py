#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:32:06 2024

@author: tusker
"""

import random
from math import pi
import time  # Import time for measuring execution time
#%%
def sample_pi(n):
    """ Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of successes."""
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s
#%%
def compute_pi(steps):
    """Computes pi and prints results"""

    ### TODO: Add/change code below
    random.seed(1)  # Serial section (seed setting)

    # Measure time for sample_pi (considered as parallelizable)
    start_parallel = time.time()
    
    n_total = steps

    s_total = sample_pi( n_total )  # Parallel section
    
    end_parallel = time.time()

    # Serial calculations (results printing and pi calculation)
    
    serial_start = time.time()
    
    pi_est = ( 4.0 * s_total ) / n_total
    
    print(" Steps\tSuccess\tPi est.\tError")
    print("%6d\t%7d\t%1.5f\t%1.5f" % ( n_total, s_total, pi_est, pi - pi_est ) )
    
    serial_end = time.time()

    # Calculate times and serial fraction
    parallel_time = end_parallel - start_parallel
    
    serial_time = ( serial_end - serial_start ) + ( serial_start - end_parallel )
    
    total_time = parallel_time + serial_time
    
    serial_fraction = serial_time / total_time

    ### TODO: Add/change code above
    return serial_fraction
#%%
if __name__ == "__main__":
    ### NOTE: The main clause will not be graded  
    ### TODO: Add/change test case to your convenience
    #steps = 1000
    steps = 1000000
    serial_fraction = compute_pi(steps)
    print(f"Serial fraction of the program: {serial_fraction:.5f}")