#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script analyses sentiment change in tweets based on precipitation data.
Usage: python global_precipitation_sentiment.py -s PATH/TO/DATASET -r PATH/TO/RESULTS -y 2022 -p 4
"""



import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from tqdm import tqdm
import xarray as xr
from glob import glob
from joblib import Parallel, delayed
import argparse



DATASET_ROOT = '/Users/rs/Harvard/Harvard Projekte/O3 Big Data Workshop/Project/data/FASRC'
RESULTS_ROOT = '/Users/rs/Harvard/Harvard Projekte/O3 Big Data Workshop/Project/data/results'
YEAR = 2022
NUM_PROCESSES = 4



# configure CLI arguments
if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-s", "--source", help="path to dataset source folder", type=str, default=DATASET_ROOT)
    argParser.add_argument("-r", "--results", help="path to results destination folder", type=str, default=RESULTS_ROOT)
    argParser.add_argument("-y", "--year", help="year to analyze", type=int, default=YEAR, choices=range(2012, 2024))
    argParser.add_argument("-p", "--processes", help="number of processes to use", type=int, default=NUM_PROCESSES, choices=range(1, 64))
    args = argParser.parse_args()

    DATASET_ROOT = args.source
    RESULTS_ROOT = args.results
    YEAR = args.year
    NUM_PROCESSES = args.processes



###############################################################################
# setup tweets table
###############################################################################
# read and pre process tweets table
tweets_df = pd.read_csv(f'{DATASET_ROOT}/twitter_sentiment_geo_index/num_posts_and_sentiment_summary_{YEAR}.csv')
tweets_df.columns = ['date', 'country', 'state', 'county', 'sentiment_score', 'tweets']
tweets_df['state'] = tweets_df.state.apply(lambda x: x.split('_')[-1])
tweets_df['county'] = tweets_df.county.apply(lambda x: x.split('_')[-1])

# read and pre process county coordinates table
county_coordinates = pd.read_csv(f'{DATASET_ROOT}/county_coordinates/lookup.csv')

# add coordinates
tweets_with_coords = tweets_df.merge(county_coordinates, on=['country', 'state', 'county'], how='left')
tweets_with_coords = tweets_with_coords.dropna(subset=['lat', 'lon'])

# full_df = tweets_with_coords.copy()
state_subset = ['Massachusetts', 'Connecticut', 'Rhode Island', 'New Hampshire', 'Vermont', 'Maine']
full_df = tweets_with_coords[tweets_with_coords.state.isin(state_subset)].copy() # test with a few close states only

# remove days with very few tweets
full_df = full_df[full_df.tweets >= 10].copy()

full_df.sentiment_score.hist(bins=100)
np.percentile(full_df.sentiment_score, 90)
print(f"loaded tweet table successfully ({len(full_df)} samples), avg sentiment score: {full_df.sentiment_score.median()}")



###############################################################################
# setup NOAA CPC data
# data source: https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html
###############################################################################
def load_dataset_for_year(year):
    filename = glob(f"{DATASET_ROOT}/precipitation/precip.{year}.nc")[0]
    return xr.open_dataset(filename)
    
def read_precipitation_values(df):
    # df = full_df.sample(100)
    dataset = load_dataset_for_year(YEAR)
    precip_values = dataset.precip.values

    df['dataset_day_idx'] = df.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday - 1)
    df['dataset_x'] = df.lon.apply(lambda x: np.abs(dataset.indexes['lon'] - (x + 360 if x < 0 else x)).argmin())
    df['dataset_y'] = df.lat.apply(lambda y: np.abs(dataset.indexes['lat'] - y).argmin())

    # precip_values = dataset.precip.values
    values_with_idxs = df.apply(lambda r: precip_values[r.dataset_day_idx, r.dataset_y, r.dataset_x], axis=1)
    return values_with_idxs



###############################################################################
# augment tweets table with PRISM precipitation data
###############################################################################
full_df['year'] = full_df.date.str[:4]

# split the df into separate chunks
df_chunks = np.array_split(full_df, NUM_PROCESSES)

start = datetime.now()

# create a list of tasks where each task is a delayed execution of the function on a chunk
delayed_function_calls = [delayed(read_precipitation_values)(chunk) for chunk in df_chunks]

# execute tasks in parallel
result_chunks = Parallel(n_jobs=NUM_PROCESSES)(delayed_function_calls)

# update the precipitation column in the original dataframe based on results
for result_subset in result_chunks:
    full_df.loc[result_subset.index, 'precipitation'] = result_subset.values

end = datetime.now()
time_delta = end - start
print(f'Augmenting the tweets table with precipitation data took {time_delta} ({NUM_PROCESSES} cores)')

# remove nan values
full_df = full_df.dropna(subset=['precipitation'])



###############################################################################
# analyze results I - absolute comparison per country
###############################################################################
RELEVANT_PRECIPITATION_THRESHOLD = 12 * 2.5 # >= 12h of at least 2.5mm/h
# df = full_df[full_df.country == 'Switzerland'].copy()

def compute_statistics(df):
    no_rain_df = df[df.precipitation == 0]
    rain_df = df[df.precipitation >= RELEVANT_PRECIPITATION_THRESHOLD]

    # descriptive groups stats
    no_rain_mean = no_rain_df.sentiment_score.mean()
    no_rain_std = no_rain_df.sentiment_score.std()

    rain_mean = rain_df.sentiment_score.mean()
    rain_std = rain_df.sentiment_score.std()

    # group sizes
    no_rain_count = len(no_rain_df)
    rain_count = len(rain_df)

    # group differences in percent
    group_diff = (no_rain_mean - rain_mean) * 100

    # effect size for comparing the means of two groups
    cohens_d = (no_rain_mean - rain_mean) / np.sqrt((no_rain_std ** 2 + rain_std ** 2) / 2)

    # Welch's t-test; statistical difference test that doesn't assume equal variances
    statistic, p_val = stats.ttest_ind(no_rain_df.sentiment_score, rain_df.sentiment_score, equal_var=False)

    return no_rain_mean, rain_mean, group_diff, no_rain_std, rain_std, cohens_d, no_rain_count, rain_count, statistic, p_val


# create a new dataframe, grouped by country
rain_no_rain_differences = full_df.groupby('country').sentiment_score.mean()
rain_no_rain_differences = rain_no_rain_differences.to_frame()
rain_no_rain_differences.columns = ['mean_sentiment_score']

# add statistical metrics
grouped_results = full_df.groupby('country').apply(compute_statistics) # TODO: this should be computed per county, and then averaged per country
nr_mean, r_mean, group_diff, nr_std, r_std, cohens_d, nr_count, r_count, statistic, p_val = zip(*grouped_results)

rain_no_rain_differences['group_diff'] = group_diff
rain_no_rain_differences['no_rain_mean'] = nr_mean
rain_no_rain_differences['rain_mean'] = r_mean

rain_no_rain_differences['no_rain_std'] = nr_std
rain_no_rain_differences['rain_std'] = r_std

rain_no_rain_differences['no_rain_days'] = nr_count
rain_no_rain_differences['rain_days'] = r_count

rain_no_rain_differences['effect_size_d'] = cohens_d
rain_no_rain_differences['p_welch'] = p_val
rain_no_rain_differences['t_welch'] = statistic

# save results
rain_no_rain_differences.to_csv(f'{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv', index=True)
print(f"saved results I successfully to '{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv'")



###############################################################################
# analyze results II - three days rain in a row
# identify instances where it rained for three consecutive days and
# compare sentiment scores of those places on the third day to places
# that had no rain for three days
###############################################################################
RELEVANT_PRECIPITATION_THRESHOLD = 5 * 2.5 # >= 5h of at least 2.5mm/h

def check_consecutive_days(group):
    # group = full_df[full_df.county == full_df.county.unique()[533]].copy()

    # copy original index for later use
    group = group.reset_index(drop=False).rename(columns={'index': 'original_index'})

    group['rainy'] = group['precipitation'] > RELEVANT_PRECIPITATION_THRESHOLD
    group['three_days_rain'] = False
    group['three_days_no_rain'] = False
    
    for i in group.index[2:]:
        if (group.loc[i, 'date'] - group.loc[i-2, 'date']).days == 2:
            if all(group.loc[i-2:i, 'rainy']):
                group.loc[i, 'three_days_rain'] = True
            if all(~group.loc[i-2:i, 'rainy']):
                group.loc[i, 'three_days_no_rain'] = True

    # return the results with the original index using 'original_index'
    three_days_rain_series = pd.Series(data=group['three_days_rain'].values, index=group['original_index'])
    three_days_no_rain_series = pd.Series(data=group['three_days_no_rain'].values, index=group['original_index'])

    return three_days_rain_series, three_days_no_rain_series

start = datetime.now()

# Convert 'date' column to datetime format
full_df['date'] = pd.to_datetime(full_df['date'])

# Sort the data
full_df = full_df.sort_values(['country', 'state', 'county', 'date'])

# split the df into separate chunks; the samples of a county need to stay together
df_chunks = [group for _, group in full_df.groupby(['country', 'state', 'county'])]

# create a list of tasks where each task is a delayed execution of the function on a chunk
delayed_function_calls = [delayed(check_consecutive_days)(chunk) for chunk in df_chunks]

# execute tasks in parallel
result_chunks = Parallel(n_jobs=NUM_PROCESSES)(delayed_function_calls)

# update the precipitation column in the original dataframe based on results
for result_subset in tqdm(result_chunks):
    three_days_rain_series = result_subset[0]
    three_days_no_rain_series = result_subset[1]
    full_df.loc[three_days_rain_series.index, 'three_days_rain'] = three_days_rain_series.values
    full_df.loc[three_days_no_rain_series.index, 'three_days_no_rain'] = three_days_no_rain_series.values

end = datetime.now()
time_delta = end - start
print(f'results II: splitting, computing, and rearranging the samples took {time_delta} ({NUM_PROCESSES} cores)')

# compute statistics & create results df
def compute_three_day_statistics(df):
    # df = full_df[full_df.country == 'Switzerland'].copy()
    
    # filter rows for the third day in both scenarios
    no_rain_sentiments = df[df['three_days_no_rain']].sentiment_score
    rainy_sentiments = df[df['three_days_rain']].sentiment_score

    # descriptive groups stats
    no_rain_mean = no_rain_sentiments.mean()
    no_rain_std = no_rain_sentiments.std()

    rain_mean = rainy_sentiments.mean()
    rain_std = rainy_sentiments.std()

    # group sizes
    no_rain_count = len(no_rain_sentiments)
    rain_count = len(rainy_sentiments)

    # group differences in percent
    group_diff = (no_rain_mean - rain_mean) * 100

    # effect size for comparing the means of two groups
    cohens_d = (no_rain_mean - rain_mean) / np.sqrt((no_rain_std ** 2 + rain_std ** 2) / 2)

    # Welch's t-test; statistical difference test that doesn't assume equal variances
    statistic, p_val = stats.ttest_ind(no_rain_sentiments, rainy_sentiments, equal_var=False)

    return no_rain_mean, rain_mean, group_diff, no_rain_std, rain_std, cohens_d, no_rain_count, rain_count, statistic, p_val


# create a new dataframe, grouped by country
three_day_difference = full_df.groupby('country').sentiment_score.mean()
three_day_difference = three_day_difference.to_frame()
three_day_difference.columns = ['mean_sentiment_score']

# add statistical metrics
grouped_results = full_df.groupby('country').apply(compute_three_day_statistics)
nr_mean, r_mean, group_diff, nr_std, r_std, cohens_d, nr_count, r_count, statistic, p_val = zip(*grouped_results)

three_day_difference['group_diff'] = group_diff
three_day_difference['no_rain_mean'] = nr_mean
three_day_difference['three_day_rain_mean'] = r_mean

three_day_difference['no_rain_std'] = nr_std
three_day_difference['three_day_rain_std'] = r_std

three_day_difference['no_rain_days'] = nr_count
three_day_difference['three_day_rain_days'] = r_count

three_day_difference['effect_size_d'] = cohens_d
three_day_difference['p_welch'] = p_val
three_day_difference['t_welch'] = statistic

# save results
three_day_difference.to_csv(f'{RESULTS_ROOT}/{YEAR}_three_day_difference.csv', index=True)
print(f"saved results II successfully to '{RESULTS_ROOT}/{YEAR}_three_day_difference.csv'")
