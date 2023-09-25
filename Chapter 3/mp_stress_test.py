#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

Produces load using multiple processes.
CAUTION: This is for demo purposes only. Do not run on a production system, as this will waste resources.
Usage: python mp_stress_test.py
"""



from multiprocessing import Pool
import time
import os



os.system("taskset -p 0xff %d" % os.getpid())

STRESS_MINS = 0.2
CORES = 4

def f(x):
    set_time = STRESS_MINS
    timeout = time.time() + 60*float(set_time)  # X minutes from now
    while True:
        if time.time() > timeout:
            break
        x*x

if __name__ == '__main__':
    print ('utilizing %d cores\n' % CORES)
    pool = Pool(CORES)
    pool.map(f, range(CORES))