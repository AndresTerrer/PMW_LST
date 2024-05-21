"""
Keep the functions needed for training here
"""
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from sklearn.model_selection import train_test_split

from datetime import datetime

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

def transform_batch(batch: pd.DataFrame):

    # Select desired dvars:
    dvars = ["tbtoa","surtep_ERA5","lat","lon","time"]
    batch = batch[dvars]

    # Remove missing data
    batch = batch.dropna()

    # We want the day_number value as an input
    batch.reset_index(inplace=True)

    # Transform the variables time, lon and day
    global_bias = (datetime(2017, 1, 1, 0, 0, 0) - datetime(2000, 1, 1, 0, 0)).total_seconds()
    batch["time"] += - global_bias - (batch["day_number"] - 1)* 24 * 60 * 60

    batch["time"] = batch["time"].apply(
        lambda x: np.sin(2 * np.pi * x / (24 * 60 * 60))
    )
    
    # Lon and lat transformations, to have a number between -1 and 1
    batch["lon"] = batch["lon"].apply(lambda x: np.sin(np.deg2rad(x)))
    batch["lat"] = batch["lat"].apply(lambda x: np.cos(np.deg2rad(x)))

    # pivot the tbtoa and time columns into 4 and 2 respectivelly
    #TODO: Can this be done within the xr.Dataset itself ??
    
    # Create new tbtoa columns based on frequency_band and polarization
    batch['tbtoa_18Ghz_V'] = batch.apply(lambda row: row['tbtoa'] if row['frequency_band'] == 0 and row['polarization'] == 0 else None, axis=1)
    batch['tbtoa_18Ghz_H'] = batch.apply(lambda row: row['tbtoa'] if row['frequency_band'] == 0 and row['polarization'] == 1 else None, axis=1)
    batch['tbtoa_37Ghz_V'] = batch.apply(lambda row: row['tbtoa'] if row['frequency_band'] == 1 and row['polarization'] == 0 else None, axis=1)
    batch['tbtoa_37Ghz_H'] = batch.apply(lambda row: row['tbtoa'] if row['frequency_band'] == 1 and row['polarization'] == 1 else None, axis=1)

    # Create new time columns based on frequency_band
    batch['time_18Ghz'] = batch.apply(lambda row: row['time'] if row['frequency_band'] == 0 else None, axis=1)
    batch['time_37Ghz'] = batch.apply(lambda row: row['time'] if row['frequency_band'] == 1 else None, axis=1)

    # Forward fill the new columns to fill None values
    batch[
        [
            'tbtoa_18Ghz_V',
            'tbtoa_18Ghz_H',
            'tbtoa_37Ghz_V',
            'tbtoa_37Ghz_H',
            'time_18Ghz',
            'time_37Ghz'
        ]
    ] = batch[
        [
            'tbtoa_18Ghz_V',
            'tbtoa_18Ghz_H',
            'tbtoa_37Ghz_V',
            'tbtoa_37Ghz_H',
            'time_18Ghz',
            'time_37Ghz'
        ]
    ].ffill()

    # Drop duplicate rows if necessary
    batch = batch.drop_duplicates(subset=['day_number', 'longitude_grid', 'latitude_grid', 'frequency_band', 'polarization'])
    batch.dropna(inplace=True)

    # Remove unwanted columns
    batch.drop(columns=['tbtoa', 'time',"latitude_grid","longitude_grid","polarization","frequency_band"], inplace=True)

    return batch


def xy_split(batch:pd.DataFrame):
    y_column = "surtep_ERA5"

    X = batch[[col for col in batch.columns if col != y_column]]
    y = batch[y_column]

    return X ,y
def default_model() -> Sequential:
    """ 
    Create a keras.model object with this architecture
    """

    n_vars = 9

    model = Sequential([
        Input((n_vars,)),
        BatchNormalization(),
        Dense(30,activation="relu", name = "hiddenLayer1"),
        Dense(20,activation="relu", name = "hiddenLayer2"),
        Dense(10,activation="relu", name = "hiddenLayer3"),
        Dense(1,activation="relu", name = "outputLayer")
    ])

    model.compile(
        optimizer = "adam",
        loss ="mse",
        metrics = ["mse"]
    )

    return model

def plot_history(history: History):
    """ 
    Standard plot of training and validation loss
    """

    fig, ax = plt.subplots(2,1, figsize = (24,10))

    ax[0].plot(history.history["loss"][2:], alpha=0.8, label = "training")
    ax[0].plot(history.history["val_loss"][2:],  alpha=0.8, label = "validation")
    ax[0].legend()
    ax[0].set_ylabel("mse [K]")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(axis="y")

    ax[1].plot(history.history["loss"], alpha=0.8, label = "training")
    ax[1].plot(history.history["val_loss"],  alpha=0.8, label = "test")
    ax[1].legend()
    ax[1].set_yscale("log")
    ax[1].set_ylabel("log_10(mse [K])")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(axis="y")

    return fig, ax

#TODO: implement this in a smarter way so I dont need to pass the model.
def training_step(model: Sequential, training_batch: pd.DataFrame, history: History=None, ) -> History :
    """ 
    Single training step with a dataframe 2000 samples long. returned expanded history
    """
    X, y = xy_split(training_batch)
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 13)
    batch_history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test,y_test))

    # Manage the history of each training run
    if history is None:
        history = batch_history
    else:
        for key in history.history.keys():
            history.history[key].extend(batch_history.history[key])

    return history


def append_training_history(full_history: History, new_history:History) -> History:
    """ 
    Extend all the keys in a model training history object.
    """
    if full_history is None:
        full_history = new_history
    else:
        for key in full_history.history.keys():
            full_history.history[key].extend(new_history.history[key])

    return full_history

