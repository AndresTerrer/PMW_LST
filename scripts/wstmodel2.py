"""
Train a simple NN with the windsat data AND TELSEM emissivity atlas (converted to netcdf)

Intended to be ported into the server and run with a full year
"""

import pickle
import os
import xarray as xr
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import windsat_datacube, model_preprocess, create_landmask
from src.model import transform_batch, xy_split

# OS Params
params = ArgumentParser()
params.add_argument(
    "--source_folder", default="./data/raw/Daily_Windsat/", help= "Folder with Windsat dataset."
)

params.add_argument(
    "--output_folder", default= "./models/", help= "Folder to store final weights and trining history."
)

params.add_argument(
    "--telsem_folder", default= "./data/processed/WinsatEmiss/", help= "Folder with the TELSEM emissivities in .nc format"
)

def build_model(n_vars: int, info:bool = False):
    model = Sequential([
        Input((n_vars,)),
        BatchNormalization(),
        Dense(30,activation="linear", name = "hiddenLayer1"),
        Dense(20,activation="relu", name = "hiddenLayer2"),
        Dense(10,activation="relu", name = "hiddenLayer3"),
        Dense(1,activation="relu", name = "outputLayer")
    ])
    model.compile(
        optimizer = Adam(learning_rate=5e-4),
        loss ="mse",
        metrics = ["mse"]
    )

    if info:
        model.summary()

    return model


if __name__ == "__main__":

    args = params.parse_args()

    folder_path = args.source_folder
    output_folder = args.output_folder
    telsem_folder = args.telsem_folder

    # Load the Emissivity dataset 
    print("Loading TELSEM atlas")
    names = os.listdir(telsem_folder)

    paths = [os.path.join(telsem_folder,name) for name in names]
    # Preprocessing of TELSEM atlas:
    telsem_ds = xr.open_mfdataset(
        paths = paths,
        engine="netcdf4",
        concat_dim="month",
        combine="nested"
    )

    # Select only the desired data variables:
    d_vars = [
        "Emis19V",
        "Emis19H",
        "Emis37V",
        "Emis37H",
    ]
    telsem_ds = telsem_ds[d_vars]

    #roll the longitude to align the data
    telsem_ds = telsem_ds.roll(
        {
            "longitude_grid" : 4 * 180
        }
    )

    landmask = create_landmask(lat = telsem_ds.lat.values, lon= telsem_ds.lon.values)
    telsem_ds["landmask"] = (("latitude_grid","longitude_grid"),landmask.values)

    telsem_ds = telsem_ds.where(telsem_ds.landmask == 0)
    telsem_ds = telsem_ds.drop_vars("landmask")
    telsem_ds = telsem_ds.reset_coords()

    # Create the telsem dataframe
    telsem_df = telsem_ds.to_dataframe().dropna().reset_index("month")

    # Map the day of the year (day_number) into the month:
    day_mapping = []
    days_in_months = [31,29,31,30,31,30,31,31,30,31,30,31]

    for i, n in enumerate(days_in_months):
        to_add = [i +1] * n
        day_mapping.extend(to_add)

    print("--- Windsat datacube model training --- ")

    #Load the dataset from the folder
    print(f"Loading windsat Datacube from {folder_path}")
    ds = windsat_datacube(folder_path)

    print("Processing data ...")
    ascds = model_preprocess(ds)
    d_vars = [
    "surtep_ERA5",
    "lat",
    "lon",
    "tbtoa_18Ghz_V",
    "tbtoa_18Ghz_H",
    "tbtoa_37Ghz_V",
    "tbtoa_37Ghz_H",
    ]

    ascds = ascds[d_vars]
    ascds_df = ascds.to_dataframe().dropna().reset_index("day_number")

    # Map day to month
    ascds_df["month"] = ascds_df["day_number"].apply(lambda x: day_mapping[x])

    # drop the day_number column
    asc_df = ascds_df.drop(columns="day_number")

    # Inner join the telsem df and the ascending df
    combined_df = pd.merge(left=asc_df, right=telsem_df, how="inner")

    # Drop the month column
    combined_df = combined_df.drop(columns="month")

    # Transform lat and lon to be periodic functions
    combined_df["lon"] = combined_df["lon"].apply(lambda x: np.sin(np.deg2rad(x)))
    combined_df["lat"] = combined_df["lat"].apply(lambda x: np.sin(np.deg2rad(x)))

    print(f"Training variables:")
    message = " -> "
    for col in combined_df.columns:
        message += col + " | "
    print(message)

    model = build_model(n_vars = len(combined_df.columns) - 1, info=True)

    # Pick the columns for training and test
    X, y = xy_split(combined_df, y_column= "surtep_ERA5")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 13)

    # Callbacks
    callback = EarlyStopping(
        monitor = "loss",
        patience = 50,
        min_delta = 0.01,
        verbose=2,
        restore_best_weights = True
    )
    checkpoints = ModelCheckpoint(
        filepath = os.path.join(output_folder, "checkpoint.keras"),
        verbose = 1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size = 512,
        validation_data=(x_test,y_test),
        callbacks=[callback, checkpoints],
        verbose = 2
    )

    now = datetime.now().strftime(r"%Y_%m_%dT%H%M%S")
    # Save the model.
    model_path = os.path.join(output_folder, f"WSMv2_{now}.keras")
    save_model(model, model_path)
    print(f"Training done, model saved as {model_path} ")

    # Save the training history:
    history_path = os.path.join(output_folder,f"WSMv2_{now}_history")
    with open(history_path,"wb") as hfile:
        pickle.dump(history.history, hfile)

    print(f"Training done, model saved as {history_path} ")


