#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script analyses sentiment change in tweets based on precipitation data.
Usage: python global_precipitation_sentiment.py
If you use the argparse version, you can use it like: python global_precipitation_sentiment.py -s PATH/TO/DATASET -r PATH/TO/RESULTS -y 2022 -p 4
"""



import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import xarray as xr
from joblib import Parallel, delayed
# import argparse # optional, for CLI arguments



DATASET_ROOT = 'PATH/TO/DATASET'
RESULTS_ROOT = 'PATH/TO/RESULTS'
YEAR = 2022
NUM_PROCESSES = 7



# # configure CLI arguments
# if __name__ == '__main__':
#     argParser = argparse.ArgumentParser()
#     argParser.add_argument("-s", "--source", help="path to dataset source folder", type=str, default=DATASET_ROOT)
#     argParser.add_argument("-r", "--results", help="path to results destination folder", type=str, default=RESULTS_ROOT)
#     argParser.add_argument("-y", "--year", help="year to analyze", type=int, default=YEAR, choices=range(2012, 2024))
#     argParser.add_argument("-p", "--processes", help="number of processes to use", type=int, default=NUM_PROCESSES, choices=range(1, 64))
#     args = argParser.parse_args()

#     DATASET_ROOT = args.source
#     RESULTS_ROOT = args.results
#     YEAR = args.year
#     NUM_PROCESSES = args.processes



###############################################################################
# setup tweets table
###############################################################################
start_ts = datetime.now()

# define a function that estimates the number of rows in a csv file
def estimate_csv_size(file_path):
    chunk = 1024 * 1024 * 10   # Process 1 MB at a time.
    f = np.memmap(file_path)
    num_newlines = sum(np.sum(f[i:i+chunk] == ord('\n')) for i in range(0, len(f), chunk))
    del f
    return num_newlines

# define data set path
tweets_file_path = f'{DATASET_ROOT}/twitter_sentiment_geo_index/num_posts_and_sentiment_summary_{YEAR}.csv'

# estimate the number of rows in the csv file (to distribute the work load across processes)
expected_rows = estimate_csv_size(tweets_file_path)
rows_per_process = expected_rows // NUM_PROCESSES

# load the county coordinates table to be used in the parallel execution
county_coordinates = pd.read_csv(f'{DATASET_ROOT}/county_coordinates/lookup.csv')

# define a function that processes a subset of the tweets table
def process_df(process_no):
    # read only a subset of the entire df
    df = pd.read_csv(
        tweets_file_path,
        sep=',',
        header=None,
        skiprows=(rows_per_process * process_no) + 1,
        nrows=rows_per_process,
        dtype={0: 'str', 1: 'str', 2: 'str', 3: 'str', 4: 'float64', 5: 'Int64'}
    )

    df.columns = ['date', 'country', 'state', 'county', 'sentiment_score', 'tweets'] # rename columns
    df['state'] = df.state.apply(lambda x: x.split('_')[-1]) # filter state names
    df['county'] = df.county.apply(lambda x: x.split('_')[-1]) # filter county names

    # add coordinates
    df = df.merge(county_coordinates, on=['country', 'state', 'county'], how='left')
    
    return df.dropna(subset=['lat', 'lon'])

# define the tasks to be executed in parallel
delayed_function_calls = [delayed(process_df)(process_no) for process_no in range(NUM_PROCESSES)]

# execute tasks in parallel
result_chunks = Parallel(n_jobs=NUM_PROCESSES)(delayed_function_calls)

# combine results
full_df = pd.concat(result_chunks)

# remove days with very few tweets
full_df = full_df[full_df.tweets >= 10]
full_df = full_df.reset_index(drop=True)

# print status
time_delta = datetime.now() - start_ts
print(f'loading and processing the twitter DF ({len(full_df)} rows) took {time_delta}sec (using {NUM_PROCESSES} processes)')



###############################################################################
# setup NOAA CPC data
# data source: https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html
###############################################################################
noaa_cpc_dataset = xr.open_dataset(f"{DATASET_ROOT}/precipitation/precip.{YEAR}.nc")

# print status
print(f"loaded NOAA CPC precipitation data successfully")

# augmentation function for the tweets table (subsets)
def augment_precipitation_values(df, lon_values, lat_values, precip_values):
    # cache indexes for faster access
    day_memory = {}
    lon_memory = {}
    lat_memory = {}

    def precipitation_for_row(row): #, lon_values, lat_values, precip_values):
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
            y = np.abs(lat_values - row.lat).argmin()
            lat_memory[row.lat] = y

        # read the precipitation value using the three computed indexes
        return precip_values[day_idx, y, x]

    # augment tweets table with the NOAA CPC precipitation data
    return df.apply(lambda row: precipitation_for_row(row), axis=1)



###############################################################################
# augment tweets table with PRISM precipitation data
###############################################################################
start_ts = datetime.now()

# prepare data to be executed in parallel; these datasets will be provided to each worker
lon_values = noaa_cpc_dataset.indexes['lon']
lat_values = noaa_cpc_dataset.indexes['lat']
precip_values = noaa_cpc_dataset.precip.values

# sort the df by country, state, county
# since each worker keeps track of the already computed day and lat/lon indixes,
# we want to try to keep similar addresses on the same worker
full_df.sort_values(['country', 'state', 'county'], inplace=True)

# split the df into separate chunks
df_chunks = np.array_split(full_df, NUM_PROCESSES)

# create a list of tasks where each task is a delayed execution of the function on a chunk
delayed_function_calls = [delayed(augment_precipitation_values)(chunk, lon_values, lat_values, precip_values) for chunk in df_chunks]

# execute tasks in parallel
result_chunks = Parallel(n_jobs=NUM_PROCESSES, prefer="processes")(delayed_function_calls)

# update the precipitation column in the original dataframe based on results
for result_subset in result_chunks:
    full_df.loc[result_subset.index, 'precipitation'] = result_subset.values

# print status
end_ts = datetime.now()
time_delta = end_ts - start_ts
print(f'Augmenting the tweets table with precipitation data took {time_delta} ({NUM_PROCESSES} cores)')
    
# remove nan values
full_df = full_df.dropna(subset=['precipitation'])



###############################################################################
# analyze results I - absolute comparison per country
###############################################################################
RELEVANT_PRECIPITATION_THRESHOLD = 12 * 2.5 # >= 12h of at least 2.5mm/h

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

# group the DF and compute statistics
grouped_statistics = (
    full_df
    .groupby(['country', 'state', 'county'])
    .apply(compute_statistics)
)

# create a summary DF
summary_df = pd.DataFrame(grouped_statistics.tolist(), columns=['no_rain_mean', 'rain_mean', 'group_diff', 'no_rain_std', 'rain_std', 'cohens_d', 'no_rain_count', 'rain_count', 'statistic', 'p_val'], index=grouped_statistics.index)

# save results
summary_df.to_csv(f'{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv', index=True)
print(f"saved results I successfully to '{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv'")

# # compute the average group difference by state & save results
relevant_results = summary_df[(summary_df.rain_count >= 20) & (summary_df.p_val < 0.05)]
average_group_diff_by_state = relevant_results.groupby('country').group_diff.mean()
# average_group_diff_by_state.to_csv(f'{RESULTS_ROOT}/{YEAR}_rain_no_rain_differences.csv', index=True)



###############################################################################
# analyze results II - three days rain in a row
# identify instances where it rained for three consecutive days and
# compare sentiment scores of those places on the third day to places
# that had no rain for three days
###############################################################################
RELEVANT_PRECIPITATION_THRESHOLD = 5 * 2.5 # >= 5h of at least 2.5mm/h

# define a function that checks if it rained for three consecutive days
def check_consecutive_days(group):
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

# prepare data to be executed in parallel
full_df['date'] = pd.to_datetime(full_df['date'])
full_df = full_df.sort_values(['country', 'state', 'county', 'date'])
df_chunks = [group for _, group in full_df.groupby(['country', 'state', 'county'])]

# start parallel execution
start = datetime.now()
result_chunks = Parallel(n_jobs=NUM_PROCESSES)(delayed(check_consecutive_days)(chunk) for chunk in df_chunks)

# update the precipitation column in the original dataframe based on results
for three_days_rain_series, three_days_no_rain_series in result_chunks:
    full_df.loc[three_days_rain_series.index, 'three_days_rain'] = three_days_rain_series.values
    full_df.loc[three_days_no_rain_series.index, 'three_days_no_rain'] = three_days_no_rain_series.values

print(f"Time elapsed: {datetime.now() - start} ({NUM_PROCESSES} cores)")

# compute statistics & create results df
def compute_three_day_statistics(df):
    no_rain_sentiments = df[df['three_days_no_rain']].sentiment_score
    rainy_sentiments = df[df['three_days_rain']].sentiment_score
    
    no_rain_mean, no_rain_std, no_rain_count = no_rain_sentiments.mean(), no_rain_sentiments.std(), len(no_rain_sentiments)
    rain_mean, rain_std, rain_count = rainy_sentiments.mean(), rainy_sentiments.std(), len(rainy_sentiments)
    group_diff = (no_rain_mean - rain_mean) * 100
    cohens_d = (no_rain_mean - rain_mean) / np.sqrt((no_rain_std ** 2 + rain_std ** 2) / 2)
    statistic, p_val = stats.ttest_ind(no_rain_sentiments, rainy_sentiments, equal_var=False)

    return no_rain_mean, rain_mean, group_diff, no_rain_std, rain_std, cohens_d, no_rain_count, rain_count, statistic, p_val

# create a new dataframe, grouped by country
grouped_results = full_df.groupby('country').apply(compute_three_day_statistics)
result_cols = ['no_rain_mean', 'three_day_rain_mean', 'group_diff', 'no_rain_std', 'three_day_rain_std', 
               'effect_size_d', 'no_rain_days', 'three_day_rain_days', 't_welch', 'p_welch']
results_df = pd.DataFrame(grouped_results.tolist(), columns=result_cols, index=grouped_results.index)

three_day_difference = full_df.groupby('country').sentiment_score.mean().reset_index(name='mean_sentiment_score')
three_day_difference = three_day_difference.set_index('country')
three_day_difference = three_day_difference.join(results_df)

# remove all incomplete rows
three_day_difference_clean = three_day_difference.dropna()

# save results
three_day_difference.to_csv(f'{RESULTS_ROOT}/{YEAR}_three_day_difference.csv', index=True)
print(f"Results saved to '{RESULTS_ROOT}/{YEAR}_three_day_difference.csv'")
