import zipfile
import io
import xarray as xr
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


def holmes(brightness_temp: xr.Dataset) -> xr.Dataset:
    """
    Calculates the surface temperature from the brightness temperature using Holmes formula
    only valid for Tb (V) and avobe 259.8 K (Brightness Temperature)
    :param brightness_temp:
    :return:
    """
    a = 1.11
    b = -15.2

    return a * brightness_temp + b


def apply_scaling(xarray_dataset: xr.Dataset) -> xr.Dataset:
    """
    Apply scaling factors to data variables in an Xarray Dataset based on the "SCALE FACTOR" attribute.

    :param xarray_dataset: The input Xarray Dataset.
    :return: A new Xarray Dataset with scaled variables.
    """
    scaled_dataset = (
        xarray_dataset.copy()
    )  # Create a copy to avoid modifying the original dataset
    print("Applying scaling")
    for dvar in tqdm(scaled_dataset.data_vars):
        if "SCALE FACTOR" in scaled_dataset[dvar].attrs.keys():
            scale_factor = scaled_dataset[dvar].attrs["SCALE FACTOR"]
            if scale_factor != 1:
                scaled_dataset[dvar] *= scale_factor
                scaled_dataset[dvar].attrs["SCALE FACTOR"] = 1  # Update the attribute

    return scaled_dataset


def load_zip(path) -> xr.Dataset:
    """
    Creates a Xarray Dataset form a zip file, located in the given path string.
    also adds a passing label for equatorial ascending and descending passes
    :params path: str
    :return: xr.Dataset
    """
    # Open the ZIP archive and list the file names
    with zipfile.ZipFile(path, "r") as zip_file:
        file_list = zip_file.namelist()

        # Initialize an empty list to store Xarray datasets
        xarray_datasets = []

        # Regular expression pattern to match the date in the file name
        date_pattern = r"_(\d{8})_\d{2}"

        # Initialize a variable for the default date
        default_date = datetime(
            2016, 12, 31
        )  # Set to one day earlier than the first valid date
        print(f"Loading zipfile from {path}")
        # Iterate through the files in the ZIP archive
        for file_name in tqdm(file_list):
            # Check if the file has an .h5 extension
            if file_name.endswith(".h5"):
                # Extract the HDF5 file
                with zip_file.open(file_name) as h5_file:
                    # Read the HDF5 file data
                    h5_data = h5_file.read()

                # Load the HDF5 file data into an Xarray dataset
                h5_data_io = io.BytesIO(h5_data)
                h5_xarray = xr.open_dataset(h5_data_io)

                # Extract the date from the file name using the regex pattern
                date_match = re.search(date_pattern, file_name)
                if date_match:
                    date_str = date_match.group(1)
                    # Convert the date string to a datetime object
                    try:
                        date = datetime.strptime(date_str, "%Y%m%d")
                    except ValueError:
                        # If an invalid date is found, use the default date
                        date = default_date
                        # We eliminate the two first days, since they are not valid.
                        continue

                    # Determine the class from the file name (EQMA or EQMD)
                    file_class = "Ascending" if "EQMA" in file_name else "Descending"

                    # Add the 'time' and 'class' coordinates to the dataset
                    h5_xarray["time"] = date
                    h5_xarray["Pass"] = file_class

                    # Append the dataset to the list
                    xarray_datasets.append(h5_xarray)

        # Concatenate the list of datasets
        xarray_dataset = xr.concat(xarray_datasets, dim="time")

    # Now, xarray_dataset contains all data with 'time' and 'class' coordinates to distinguish between EQMA and EQMD
    xarray_dataset = xarray_dataset.set_coords("Pass")

    return xarray_dataset


def extract_timeseries(xarray_dataset, lat, lon) -> xr.DataArray:
    """
    Extract the timeseries from a pixel in the map for all instances of time

    :param xarray_dataset: Dataset with at least lat lon and time coordinates
    :param lat: float
    :param lon: float
    :return: np.array 1D
    """

    timeseries = np.zeros(shape=(len(xarray_dataset.time),))

    for i in range(len(xarray_dataset.time)):
        timeseries[i] = xarray_dataset[i, lat, lon]

    return np.array(timeseries)


def create_landmask(
    xarray_dataset, threshold=273, show=False, file_path=None, figure_path=None
) -> xr.DataArray:
    """
    Creates a mask for latitude and longitude given a threshold and a base array.
    Save the resulting xr.DataArray as a file if a path is provided.

    :param xarray_dataset: xr.DataSet with latitude and longitude dimensions
    :param threshold: float
    :return: xr.DataArray with latitude and longitude
    """
    landmask = xarray_dataset.where(xarray_dataset > threshold, other=0)
    landmask = landmask.where(landmask == 0, other=1)

    # Save the mask into a file option
    if file_path is not None:
        landmask.to_netcdf(file_path)
    if show:
        landmask.plot(cmap="jet")
        # Save an image of the mask option
        if figure_path is not None:
            plt.savefig(figure_path)

    return landmask
