import requests
from tqdm import tqdm



# change this variable to download the dataset you want to work with
# valid range is 2012 to 2023
YEAR = 2022

# source of the precipitation dataset
# https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html
precipitation_url = f"https://psl.noaa.gov/thredds/fileServer/Datasets/cpc_global_precip/precip.{YEAR}.nc"



def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

print(f"Downloading precipitation dataset for the year {YEAR}:")
target_filename = f"precip.{YEAR}.nc"
download(url=precipitation_url, fname=target_filename, chunk_size=1024)
