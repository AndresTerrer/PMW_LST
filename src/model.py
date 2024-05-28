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


#TODO: Rename this function, since we can use it with the whole dataset
def transform_batch(df: pd.DataFrame):
    """ 
    Numerical transformations applied to the variables in the training dataframe.

    Time == time of observation (seconds since 2000-01-01T00:00:00) transformed 
    into "day fraction" [0-1] since midnight, then wrapped into a periodic function 
    (both 0 and 1 represent midnight)

    Longitude (degrees) turned into the sin of the angle instead for the same periodic reason
    Latitude also transformed with sin(x) for consistency but it could be normalised instead
    [-90, 90] -> [-1, 1]

    Day number (Day of the Year) normalized by 366 and also periodic
    """
    batch = df.copy()
    # Transform the variables time, lon and day
    global_bias = (datetime(2017, 1, 1, 0, 0, 0) - datetime(2000, 1, 1, 0, 0)).total_seconds()
    batch["time_18Ghz"] += - global_bias - (batch["day_number"] - 1)* 24 * 60 * 60

    batch["time_18Ghz"] = batch["time_18Ghz"].apply(
        lambda x: np.sin(2 * np.pi * x / (24 * 60 * 60))
    )

    global_bias = (datetime(2017, 1, 1, 0, 0, 0) - datetime(2000, 1, 1, 0, 0)).total_seconds()
    batch["time_37Ghz"] += - global_bias - (batch["day_number"] - 1)* 24 * 60 * 60

    batch["time_37Ghz"] = batch["time_37Ghz"].apply(
        lambda x: np.sin(2 * np.pi * x / (24 * 60 * 60))
    )
    
    # Lon and lat transformations, to have a number between -1 and 1
    batch["lon"] = batch["lon"].apply(lambda x: np.sin(np.deg2rad(x)))
    batch["lat"] = batch["lat"].apply(lambda x: np.sin(np.deg2rad(x)))

    # Day of the year is also a periodic qualifyer
    batch["day_number"] = batch["day_number"].apply(lambda x: np.sin(2 * np.pi * x / 366))

    return batch


def xy_split(batch:pd.DataFrame, y_column: str = "surtep_ERA5"):
    """ 
    Split the training dataset into variables for prediction and true value to predict
    """
    X = batch[[col for col in batch.columns if col != y_column]]
    y = batch[y_column]

    return X ,y

# TODO: move this into its own thing. Random search of the architecture
def default_model() -> Sequential:
    """ 
    UNUSED at the moment.
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

def plot_history(history: dict):
    """ 
    Standard plot of training and validation loss
    """

    fig, ax = plt.subplots(1,2, figsize = (24,10))

    ax[0].plot(history["loss"], alpha=0.8, label = "training")
    ax[0].plot(history["val_loss"],  alpha=0.8, label = "validation")
    ax[0].legend()
    ax[0].set_ylabel("mse [K]")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(axis="y")

    last_epochs = len(history["loss"])//2
    ax[1].plot(history["loss"][-last_epochs:], alpha=0.8, label = "training")
    ax[1].plot(history["val_loss"][-last_epochs:],  alpha=0.8, label = "validation")
    ax[1].legend()
    ax[1].set_ylabel("mse [K]")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(axis="y")
    ax[1].set_title(f"Last {last_epochs} epochs")

    return fig, ax

#TODO: implement this in a smarter way so I dont need to pass the model.
def training_step(model: Sequential, training_batch: pd.DataFrame, history: History=None, ) -> History :
    """ 
    UNUSED
    Single training step with a dataframe 2000 samples long. returned expanded history
    """
    X, y = xy_split(training_batch)
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 13)
    batch_history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test))

    # Manage the history of each training run
    if history is None:
        history = batch_history
    else:
        for key in history.history.keys():
            history.history[key].extend(batch_history.history[key])

    return history


def append_training_history(full_history: History, new_history:History) -> History:
    """ 
    UNUSED
    Extend all the keys in a model training history object.
    """
    if full_history is None:
        full_history = new_history
    else:
        for key in full_history.history.keys():
            full_history.history[key].extend(new_history.history[key])

    return full_history

