#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script analyses sentiment change in tweets based on precipitation data.
Usage: python precipitation_sentiment_toy_example_chap3.py
"""



import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr



DATASET_ROOT = 'PATH/TO/DATASET'
RESULTS_ROOT = 'PATH/TO/RESULTS'
YEAR = 2022



###############################################################################
# setup tweets table
###############################################################################
# take start time
start_execution_timestamp = datetime.now()

# read and pre process tweets table
tweets_df = pd.read_csv(f'{DATASET_ROOT}/twitter_sentiment_geo_index/num_posts_and_sentiment_summary_{YEAR}.csv')
tweets_df.columns = ['date', 'country', 'state', 'county', 'sentiment_score', 'tweets'] # rename columns
tweets_df['state'] = tweets_df.state.apply(lambda x: x.split('_')[-1]) # filter state names
tweets_df['county'] = tweets_df.county.apply(lambda x: x.split('_')[-1]) # filter county names

# define toy example with a few states only
state_subset = ['Massachusetts', 'Connecticut', 'Rhode Island']
tweets_subset = tweets_df[tweets_df.state.isin(state_subset)] # test with a few close states only

# read the county coordinates table
county_coordinates = pd.read_csv(f'{DATASET_ROOT}/county_coordinates/lookup.csv')

# add coordinates
tweets_subset = tweets_subset.merge(county_coordinates, on=['country', 'state', 'county'], how='left')
tweets_subset = tweets_subset.dropna(subset=['lat', 'lon'])

# print status
print(f"loaded tweets and county coordinates table successfully ({len(tweets_subset)} samples), avg sentiment score: {tweets_subset.sentiment_score.median()}")



###############################################################################
# setup NOAA CPC data
###############################################################################
noaa_cpc_dataset = xr.open_dataset(f"{DATASET_ROOT}/precipitation/precip.{YEAR}.nc")

# print status
print(f"loaded NOAA CPC precipitation data successfully")

day_memory = {}
lon_memory = {}
lat_memory = {}

def precipitation_for_row(row):

    # compute the array index for the day of the year
    if row.date in day_memory:
        day_idx = day_memory[row.date]
    else:
        day_idx = datetime.strptime(row.date, "%Y-%m-%d").timetuple().tm_yday - 1
        day_memory[row.date] = day_idx

    # compute array index for longitude value
    if row.lon in lon_memory:
        x = lon_memory[row.lon]
    else:
        lon_values = noaa_cpc_dataset.indexes['lon']
        if row.lon < 0:
            lon_0_to_360 = row.lon + 360
        else:
            lon_0_to_360 = row.lon
        x = np.abs(lon_values - lon_0_to_360).argmin()
        lon_memory[row.lon] = x

    # compute array index for latitude value
    if row.lat in lat_memory:
        y = lat_memory[row.lat]
    else:
        lat_values = noaa_cpc_dataset.indexes['lat']
        y = np.abs(lat_values - row.lat).argmin()
        lat_memory[row.lat] = y

    # read the precipitation value using the three computed indexes
    return noaa_cpc_dataset.precip.values[day_idx, y, x]

# augment tweets table with the NOAA CPC precipitation data
start_execution_timestamp = datetime.now()
tweets_subset['precipitation'] = tweets_subset.apply(lambda row: precipitation_for_row(row), axis=1)
end_execution_timestamp = datetime.now()
print(f"this script took {end_execution_timestamp - start_execution_timestamp} to execute")

# remove nan values
tweets_subset = tweets_subset.dropna(subset=['precipitation'])

# print status
print(f"successfully augmented tweets table with precipitation data ({len(tweets_subset)} samples)")



###############################################################################
# analyze results I - absolute comparison per country
###############################################################################
RELEVANT_PRECIPITATION_THRESHOLD = 12 * 2.5  # >= 12h of at least 2.5mm/h

def compute_statistics(grouped_df):
    no_rain_df = grouped_df[grouped_df.precipitation == 0]
    rain_df = grouped_df[grouped_df.precipitation >= RELEVANT_PRECIPITATION_THRESHOLD]
    
    no_rain_mean = no_rain_df.sentiment_score.mean()
    rain_mean = rain_df.sentiment_score.mean()
    group_diff_percent = (no_rain_mean - rain_mean) * 100
    
    return no_rain_mean, rain_mean, group_diff_percent

# group the DF and compute statistics
grouped_statistics = (
    tweets_subset
    .groupby(['country', 'state', 'county'])
    .apply(compute_statistics)
)

# create a summary DF
summary_df = pd.DataFrame(grouped_statistics.tolist(), columns=['no_rain_mean', 'rain_mean', 'group_diff'], index=grouped_statistics.index)

# compute the average group difference by state & save results
average_group_diff_by_state = summary_df.groupby('country').group_diff.mean()
average_group_diff_by_state.to_csv(f'{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv', index=True)

# take end time
end_execution_timestamp = datetime.now()

# print status
print(f"analysis complete")
print(f"this script took {end_execution_timestamp - start_execution_timestamp} to execute")
