"""
Train a simple NN with the windsat data
Intended to be ported into the server and run with a full year
"""

import pickle
import os
from argparse import ArgumentParser
from datetime import datetime
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import windsat_datacube, model_preprocess
from src.model import transform_batch, xy_split

# OS Params
params = ArgumentParser()
params.add_argument(
    "--source_folder", default="./data/raw/Daily_Windsat/", help= "Folder with Windsat dataset."
)

params.add_argument(
    "--output_folder", default= "./models/", help= "Folder to store final weights and trining history."
)

def build_model(info:bool = False):
    n_vars = ascds_df.shape[1] - 1 # dont count the prediction column.
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

    print("--- Windsat datacube model training --- ")

    #Load the dataset from the folder
    print(f"Loading windsat Datacube from {folder_path}")
    ds = windsat_datacube(folder_path)

    print("Processing data ...")
    ascds = model_preprocess(ds)

    ascds_df = ascds.to_dataframe()
    ascds_df.reset_index(inplace=True)
    ascds_df.dropna(inplace=True)
    ascds_df.drop(columns=["longitude_grid","latitude_grid"], inplace=True)
    ascds_df = transform_batch(ascds_df)

    model = build_model(info=True)

    # Pick the columns for training and test
    X, y = xy_split(ascds_df)
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 13)

    # Callbacks
    callback = EarlyStopping(
        monitor = "loss",
        patience = 2,
        min_delta = 0.05,
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
        epochs=20,
        validation_data=(x_test,y_test),
        callbacks=[callback, checkpoints],
        verbose = 2
    )

    # Save the model.
    now = datetime.now().strftime(r"%Y_%m_%dT%H%M%S")
    weights_path = os.path.join(output_folder, f"{now}.keras")
    model.save_model(weights_path)

