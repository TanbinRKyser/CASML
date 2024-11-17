#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import argparse
import random
from math import pi
from time import time
import matplotlib.pyplot as plt
from itertools import product


def sample_pi(n, seed):
    """Perform n steps of Monte Carlo simulation for estimating Pi/4.
        Returns the number of successes."""
    random.seed(seed)
    s = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s


def worker(task_queue, result_queue):
    """Worker function to process tasks from task_queue and send results to result_queue."""
    while True:
        task = task_queue.get()
        
        if task is None:  # Termination signal
            task_queue.task_done()
            break
        
        n, seed = task
        
        result = sample_pi(n, seed)
        result_queue.put(result)
        
        task_queue.task_done()


def compute_pi(accuracy, workers, seed):
    """Compute the value of Pi using Monte Carlo simulation until the desired accuracy is achieved."""
    random.seed(seed)

    steps = 10**6  # Number of steps per batch
    s_total = 0
    n_total = 0
    pi_est = 0.0

    # Create multiprocessing queues
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    # Start worker processes
    processes = []
    
    for _ in range(workers):
        p = multiprocessing.Process( target = worker, args = ( task_queue, result_queue ) )
        p.start()
        processes.append( p )

    start_time = time()
    
    try:
        while True:
            # Generate new tasks
            for i in range( workers ):
                task_queue.put( ( steps, seed + i + n_total ) )  # Unique seed per worker

            # Wait for all tasks to complete
            task_queue.join()
            
            # for i in product([n], seeds):
            #     print(i)
            
            # Collect results
            for _ in range( workers ):
                s_total += result_queue.get()

            # Update totals and estimate Pi
            n_total += steps * workers
            pi_est = ( 4.0 * s_total ) / n_total
            error = abs( pi - pi_est )

            # Log progress
            print(f"Workers: {workers}, Steps: {n_total}, Pi Estimate: {pi_est:.6f}, Error: {error:.6f}")

            # Check if desired accuracy is achieved
            if error < accuracy:
                break
    finally:
        # Send termination signals to workers
        for _ in range( workers ):
            task_queue.put( None )
        for p in processes:
            p.join()

    end_time = time()
    elapsed_time = end_time - start_time

    # Calculate samples per second
    samples_per_second = n_total / elapsed_time

    # Final results
    print("Steps\tSuccess\tPi est.\tError")
    #print(f"{n_total:6d}\t{s_total:7d}\t{pi_est:1.5f}\t{error:1.5f}")
    print( "%6d\t%7d\t%1.5f\t%1.5f" % ( n_total, s_total, pi_est, pi-pi_est ) )
    return samples_per_second


def plot_workers_vs_performance(worker_counts, samples_per_second, specified_worker, f_parallel=0.9):
    """Plot number of workers vs. samples per second, with theoretical speedup and a vertical line."""

    theoretical_speedup = [
        1 / ( ( 1 - f_parallel ) + ( f_parallel / workers ) ) for workers in worker_counts
    ]

    max_samples_per_second = max( samples_per_second )

    scaled_theoretical_speedup = [ s * max_samples_per_second for s in theoretical_speedup ]

    ## plot
    plt.figure( figsize=( 10, 6 ) )

    # Plot measured performance
    plt.plot( worker_counts, samples_per_second, marker='o', linestyle='-', label="Measured Performance")

    # Plot theoretical speedup
    plt.plot( worker_counts, scaled_theoretical_speedup, marker='x', linestyle='--', label="Theoretical Speedup (Amdahl's Law)")

    # User specified number of workers
    plt.axvline( x = specified_worker, color='red', linestyle='--', label=f"Specified Workers ({specified_worker})")

    plt.xlabel("Number of Workers")
    plt.ylabel("Samples per Second")
    plt.title("Performance: Workers vs Samples per Second")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation for Estimating Pi")
    
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        required=True,
        help="Specify the number of workers (default: 2)",
    )
    
    parser.add_argument(
        "-a",
        "--accuracy",
        type=float,
        default=0.0001,
        help="Desired accuracy for the simulation (default: 0.0001)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    args = parser.parse_args()

    # Fixed worker counts for the plot
    worker_counts = [1, 2, 4, 8, 16]
    specified_worker = args.workers
    accuracy = args.accuracy
    seed = args.seed

    performance_results = []


    for workers in worker_counts:
        print(f"\nRunning simulation with {workers} workers...")
        
        samples_per_second = compute_pi(accuracy, workers, seed)
        performance_results.append(samples_per_second)
        
        print(f"Workers: {workers}, Samples per Second: {samples_per_second:.2f}")

    # Plot results with theoretical speedup and specified worker vertical line
    plot_workers_vs_performance(worker_counts, performance_results, specified_worker, f_parallel=0.9)
