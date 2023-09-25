#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script spawn multiple processes and sleeps; the purpose is to demonstrate running multiple processes on a cluster without causing load.
Usage: python mp_sleep.py
"""



from joblib import Parallel, delayed
import time



# Number of processes to use
NUM_PROCESSES = 4
SLEEP_TIME = 300 # seconds; 5 minutes



def sleep_function():
    print("Sleeping on one core...")
    time.sleep(SLEEP_TIME)
    print("Waking up on one core...")


print("Program started")

# Run sleep_function on multiple cores
job_list = [delayed(sleep_function)() for _ in range(NUM_PROCESSES)]
Parallel(n_jobs=NUM_PROCESSES)(job_list)

print("Program ended")
