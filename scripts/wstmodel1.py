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
from tensorflow.keras.models import save_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import windsat_datacube, model_preprocess
from src.model import xy_split, create_training_df

# OS Params
params = ArgumentParser()
params.add_argument(
    "--source_folder", default="./data/raw/daily_Windsat/", help= "Folder with Windsat dataset."
)

params.add_argument(
    "--output_folder", default= "./models/", help= "Folder to store final weights and trining history."
)

# Swath selection
params.add_argument(
    "--swath_sector", default= 0 , help = "Train a model on Ascending (0, default) or Descending (1) pass data."
)

def build_model(n_vars: int , info:bool = False):
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
    swath_sector = int(args.swath_sector)

    swath2char = {
        0 : "A", # Ascensing pass (6 PM)
        1 : "D", # Descending pass (6 AM)
    }

    print("--- Windsat datacube model training --- ")

    print(f"Swath sector '{swath2char[swath_sector]}' ")

    #Load the dataset from the folder
    print(f"Loading windsat Datacube from {folder_path}")
    ds = windsat_datacube(folder_path)

    print("Processing data ...")
    ds = model_preprocess(
        ds,
        swath_sector=swath_sector,
        look="impute",
        add_look_flag=False
    )

    df = create_training_df(ds)

    print(f"Training variables:")
    message = " -> "
    for col in df.columns:
        message += col + "  "
    print(message)

    n_vars = df.shape[1] - 1
    model = build_model(n_vars, info=True)

    # Pick the columns for training and test
    X, y = xy_split(df, y_column="surtep_ERA5")
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 13)

    # Callbacks
    callback = EarlyStopping(
        monitor = "loss",
        patience = 30,
        min_delta = 0.05,
        verbose=2,
        restore_best_weights = True
    )
    checkpoints = ModelCheckpoint(
        filepath = os.path.join(output_folder, "checkpoint.keras"),
        verbose = 2
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=300,
        batch_size = 512,
        validation_data=(x_test,y_test),
        callbacks=[callback, checkpoints],
        verbose = 2
    )

    now = datetime.now().strftime(r"%Y_%m_%dT%H%M%S")
    # Save the model.
    model_path = os.path.join(
        output_folder, f"WSMv1_{swath2char[swath_sector]}_{now}.keras"
    )
    save_model(model, model_path)
    print(f"Training done, model saved as {model_path} ")

    # Save the training history:
    history_path = os.path.join(output_folder,f"WSMv1_{swath2char[swath_sector]}_{now}_history")
    with open(history_path,"wb") as hfile:
        pickle.dump(history.history, hfile)

    print(f"Training done, model saved as {history_path} ")


