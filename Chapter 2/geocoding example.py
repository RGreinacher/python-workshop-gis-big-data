#!/usr/bin/env python

"""
Harvard University, Center for Geographic Analyses
Workshop: Python for Geospatial Big Data and Data Science Using the FASRC, Sept 26th, 2023
Robert P. Spang, TU Berlin, Germany & CGA, Harvard University, USA (spang@tu-berlin.de)

This script showcases how geocoding against a service API works.
This is supplementary material for Chapter 2 of the workshop and not part of the exercise.
To run this script, create a HERE API key and replace the value of HERE_API_KEY below.
https://www.here.com/platform/geocoding

HERE Maps is only one of many geocoding services.
"""



import pandas as pd
import requests
from tqdm import tqdm



HERE_API_KEY = 'AAbbCCddEEffGGhhIIjjKKllMMnnOOppQQrrSSttUUvvWWxxYYzz' # replace this KEY with your own



###############################################################################
# setup address table
###############################################################################
example_df = pd.DataFrame({
    'country': {0: 'Brazil', 1: 'Mexico', 2: 'Mexico', 3: 'Netherlands', 4: 'Nigeria', 5: 'Sweden', 6: 'Turkey', 7: 'Uganda', 8: 'United States', 9: 'Venezuela'},
    'state': {0: 'Ceará', 1: 'Chiapas', 2: 'Veracruz', 3: 'Gelderland', 4: 'Lagos', 5: 'Skåne', 6: 'Adana', 7: 'Mbarara', 8: 'Kentucky', 9: 'Mérida'},
    'county': {0: 'Morada Nova', 1: 'Tonalá', 2: 'Oluta', 3: 'Geldermalsen', 4: 'Ojo', 5: 'Burlöv', 6: 'Tufanbeyli', 7: 'Mbarara', 8: 'Laurel', 9: 'Andrés Bello'}
})

example_df['address'] = example_df.apply(lambda x: f"{x.county}, {x.state}, {x.country}", axis=1)



###############################################################################
# geocoding: request coordinates from addresses
###############################################################################
# define the API endpoint
URL = "https://geocode.search.hereapi.com/v1/geocode"


# create a geocode lookup function
def request_coordinates(address):

    # encode the address and the API key as URL parameters
    params = {'q': f"{address}", 'apikey': HERE_API_KEY}

    try:
        # perform HTTP GET request
        data = requests.get(url = URL, params = params).json()

        # try to extract lat and lon
        position = data['items'][0]['position']
        
        # return lat, lon, indicate success, and omit the last value (error response data)
        return position['lat'], position['lng'], True, None
    
    except:
        # return no lat, no lon, indicate an error, and return the response data
        return None, None, False, data


# iterate over the data frame and provide a progress bar
for idx, row in tqdm(example_df.iterrows(), total=len(example_df)):

    # call the lookup function
    lat, lon, success, response = request_coordinates(row.address)

    # process the response
    if success:
        example_df.loc[idx, ['lat', 'lon']] = lat, lon
    else:
        if response == {'items': []}:
            print(f'WARNING: {row.address} not found')
            example_df.loc[idx, 'not_found'] = True
        else:
            print(f'ERROR: {response}')
            break
    

# save to csv
example_df.to_csv(f'address_geocode_example.csv', index=False)
