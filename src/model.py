"""
Keep the functions needed for training here
"""

from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np

def get_training_batch(index_list: list, ds: xr.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Get a pair of X and y data to train, given a list of indeces.
    ds: open windsat xarray dataset
    """
    # TODO: async version, geting the data batches and training the model should not be done in series.

    # Substract the number of seconds between the start of the year and the time origin
    global_bias = (datetime(2017,1,1,0,0,0) - datetime(2000,1,1,0,0)).total_seconds()

    batch_list = []
    for day, latg, long in index_list:
        subset = ds.sel(day_number = day +1, latitude_grid = latg, longitude_grid=long)

        time_18ghz, time_37ghz = subset.time.values.flatten()

        tbtoa_18ghz_V , tbtoa_37ghz_V, tbtoa_18ghz_H, tbtoa_37ghz_H = subset.tbtoa.values.flatten()

        # Also, this will be parallelized 
        xv ={
            # "swath": swath,
            "day_number" : day + 1,
            "lat" : float(subset.lat.values),
            "lon" : float(subset.lon.values),

            # Normalized time (seconds since midnight UTC to fraction of the day)
            "time_18ghz" : time_18ghz,
            "time_37ghz" : time_37ghz,

            "tbtoa_18ghz_V" : tbtoa_18ghz_V,
            "tbtoa_37ghz_V" : tbtoa_37ghz_V,
            "tbtoa_18ghz_H" : tbtoa_18ghz_H,
            "tbtoa_37ghz_H" : tbtoa_37ghz_H,

            # "quality_flag" : float(subset.quality_flag.values),

            # Prediction data
            "surtep_ERA5" : float(subset.surtep_ERA5.values),
            # "airtep_ERA5" : float(subset.airtep_ERA5.values),
        }   

        batch_list.append(xv)

    batch_df = pd.DataFrame(batch_list)
    #NOTE Error by 1, we need to substract 1 day, since the origin is 2017-01-01. not 2017-0-0.
    #NOTE: All times in seconds since midnight UTC
    batch_df["time_18ghz"] += - global_bias - (batch_df["day_number"] - 1)* 24 * 60 * 60
    batch_df["time_37ghz"] += - global_bias - (batch_df["day_number"] - 1)* 24 * 60 * 60

    # Normalise the values to be between 0 and 1 (0.5 = mid day) 
    batch_df["time_18ghz"] = batch_df["time_18ghz"] / (24*60*60)
    batch_df["time_37ghz"] = batch_df["time_37ghz"] / (24*60*60)

    # Loop longitude so 0 and 360 are close.
    batch_df["lon"] = batch_df["lon"].apply(lambda x: np.sin(np.deg2rad(x)))

    y_vars = ["surtep_ERA5"]

    x_train = batch_df[[col for col in batch_df.columns if col not in y_vars]]
    y_train = batch_df[y_vars]
    return x_train, y_train


