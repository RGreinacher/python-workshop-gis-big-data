#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script spawns multiple processes and sleeps; the purpose is to demonstrate running multiple processes using joblib.

Usage: python mp_sleep.py
"""



from joblib import Parallel, delayed
import time



NUM_PROCESSES = 4 # number of processes to use
SLEEP_TIME = 30 # time in seconds



def sleep_function(time_to_sleep):
    print("Sleeping on one process...")
    time.sleep(time_to_sleep)
    print("Waking up on one process...")


print("Program started")

# Run sleep_function on multiple processes
job_list = [delayed(sleep_function)(SLEEP_TIME) for _ in range(NUM_PROCESSES)]
Parallel(n_jobs=NUM_PROCESSES)(job_list)

print("Program ended")
