"""
Train a simple NN with the windsat data AND TELSEM emissivity atlas (converted to netcdf)

Intended to be ported into the server and run with a full year
"""

import pickle
import os
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

from src.processing import windsat_datacube, model_preprocess, telsem_datacube, doy2month_mapping
from src.model import xy_split

# OS Params
params = ArgumentParser()
params.add_argument(
    "--source_folder", default="./data/raw/daily_Windsat/", help= "Folder with Windsat dataset."
)

params.add_argument(
    "--output_folder", default= "./models/", help= "Folder to store final weights and trining history."
)

params.add_argument(
    "--telsem_folder", default= "./data/processed/WindsatEmiss/", help= "Folder with the TELSEM emissivities in .nc format"
)

params.add_argument(
    "--swath_sector", default=0, help = "Ascending pass = 0, Descending pass = 1."
)

def build_model(n_vars: int, info:bool = False):
    model = Sequential([
        Input((n_vars,)),
        BatchNormalization(),
        Dense(60,activation="linear", name = "hiddenLayer1"),
        Dense(30,activation="relu", name = "hiddenLayer2"),
        Dense(15,activation="relu", name = "hiddenLayer3"),
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
    swath_sector = int(args.swath_sector)

    swath2char = {
        0 : "A", # Ascensing pass (6 AM)
        1 : "D", # Descending pass (6 PM)
    }

    # Load the Emissivity dataset 
    print(f"Loading TELSEM atlas from {telsem_folder}")
    telsem_ds = telsem_datacube(telsem_folder)

    # Create the telsem dataframe
    telsem_df = telsem_ds.to_dataframe().dropna().reset_index("month")

    #Load the dataset from the folder
    print(f"Loading windsat Datacube from {folder_path}")
    ws_ds = windsat_datacube(folder_path)

    print("Processing data ...")
    swath_ds = model_preprocess(ws_ds, swath_sector=swath_sector, look="impute", add_look_flag=False)
    d_vars = [
    "surtep_ERA5",
    "lat",
    "lon",
    "tbtoa_18Ghz_V",
    "tbtoa_18Ghz_H",
    "tbtoa_37Ghz_V",
    "tbtoa_37Ghz_H",
    ]

    swath_ds = swath_ds[d_vars]
    swath_df = swath_ds.to_dataframe().dropna().reset_index("day_number")

    # Map the day of the year (day_number) into the month:
    day_mapping = doy2month_mapping()    
    swath_df["month"] = swath_df["day_number"].apply(lambda x: day_mapping[x])

    # drop the day_number column
    swath_df = swath_df.drop(columns="day_number")

    # Inner join the telsem df and the ascending df
    combined_df = pd.merge(left=swath_df, right=telsem_df, how="inner")

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

    print(f"Training for swaht sector: {swath2char[swath_sector]}")

    model = build_model(n_vars = len(combined_df.columns) - 1, info=True)

    # Pick the columns for training and test
    X, y = xy_split(combined_df, y_column= "surtep_ERA5")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 13)

    # Callbacks
    callback = EarlyStopping(
        monitor = "loss",
        patience = 100,
        min_delta = 0.01,
        verbose=2,
        restore_best_weights = True
    )
    checkpoints = ModelCheckpoint(
        filepath = os.path.join(output_folder, f"checkpoint_{swath2char[swath_sector]}.keras"),
        verbose = 1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size = 1024,
        validation_data=(x_test,y_test),
        callbacks=[callback, checkpoints],
        verbose = 2
    )

    now = datetime.now().strftime(r"%Y_%m_%dT%H%M%S")
    # Save the model.
    model_path = os.path.join(
        output_folder, f"WSMv2_{swath2char[swath_sector]}_{now}.keras"
    )
    save_model(model, model_path)
    print(f"Training done, model saved as {model_path} ")

    # Save the training history:
    history_path = os.path.join(
        output_folder,f"WSMv2_{swath2char[swath_sector]}_{now}_history"
    )
    with open(history_path,"wb") as hfile:
        pickle.dump(history.history, hfile)

    print(f"Training done, model saved as {history_path} ")


