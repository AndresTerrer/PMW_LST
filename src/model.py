"""
Keep the functions needed for training here
"""
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from sklearn.model_selection import train_test_split

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr


#TODO: Remove this function, replaced by "create_training_df"
def transform_batch(df: pd.DataFrame):
    """ 
    Numerical transformations applied to the variables in the training dataframe.

    Longitude (degrees) turned into the sin of the angle instead for the same periodic reason
    Latitude also transformed with sin(x) for consistency but it could be normalised instead
    [-90, 90] -> [-1, 1]

    """
    batch = df.copy()
    # Transform the variables lon and lat

    # Lon and lat transformations, to have a number between -1 and 1
    batch["lon"] = batch["lon"].apply(lambda x: np.sin(np.deg2rad(x)))
    batch["lat"] = batch["lat"].apply(lambda x: np.sin(np.deg2rad(x)))

    return batch

def create_training_df(ds: xr.Dataset) -> pd.DataFrame:
    """ 
    Do all the necessary manipulations to turn a dataset into a dataframe that
    can be fed to a keras model for trining.


    """
    # In built xarray method
    df = ds.to_dataframe()
    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    # remove coordinate columns
    coord_names = list(ds._coord_names)

    df.drop(columns = coord_names, inplace=True)

    # Apply trig transformations to lat and lon 

    df["lon"] = df["lon"].apply(lambda x: np.sin(np.deg2rad(x)))
    df["lat"] = df["lat"].apply(lambda x: np.sin(np.deg2rad(x)))


    return df


def xy_split(batch:pd.DataFrame, y_column: str = "surtep_ERA5"):
    """ 
    Split the training dataset into variables for prediction and true value to predict
    """
    X = batch[[col for col in batch.columns if col != y_column]]
    y = batch[y_column]

    return X ,y

def default_model(n_vars: int, info: bool = True) -> Sequential:
    """ 
    UNUSED at the moment.
    Create a keras.model object with this architecture
    """
    model = Sequential([
        Input((n_vars,)),
        BatchNormalization(),
        Dense(60,activation="linear", name = "hiddenLayer1"),
        Dense(30,activation="relu", name = "hiddenLayer2"),
        Dense(15,activation="relu", name = "hiddenLayer3"),
        Dense(1,activation="relu", name = "outputLayer")
    ])

    model.compile(
        optimizer = "adam",
        loss ="mse",
        metrics = ["mse"]
    )

    if info:
        model.summary()

    return model

def plot_history(history: dict, loss_threshold: float = None):
    """ 
    Standard plot of training and validation loss.

    param loss_threshold: split the training history so that the 
    second plot shows every epuch below this threshold. Default shows
    the last half of the training history.
    """

    fig, ax = plt.subplots(1,2, figsize = (24,10))

    ax[0].plot(history["loss"], alpha=0.8, label = "training")
    ax[0].plot(history["val_loss"],  alpha=0.8, label = "validation")
    ax[0].legend()
    ax[0].set_ylabel("log(mse [K²])")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(axis="y")
    ax[0].set_yscale("log")

    if loss_threshold is not None:
        for i, loss in enumerate(history["loss"]):
            if loss < loss_threshold:
                start_epoch = i
                break
            
            else:
                start_epoch = len(history["loss"])//2

    else: 
        start_epoch = len(history["loss"])//2

    ax[1].plot(history["loss"][start_epoch:], alpha=0.8, label = "training")
    ax[1].plot(history["val_loss"][start_epoch:],  alpha=0.8, label = "validation")
    ax[1].legend()
    ax[1].set_ylabel("mse [K²]")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(axis="y")
    ax[1].set_title(
        f"Epochs after loss < {loss_threshold}" 
        if loss_threshold is not None else 
        f"Last {len(history['loss']) - start_epoch} epochs"
    )

    return fig, ax
