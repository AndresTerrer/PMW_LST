import zipfile
import io
import xarray as xr
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from functools import partial
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


def recover_dates(folder_path: str) -> list[date]:
    """Helper, returns a list of dates parsing the files within the folder"""

    dates = []
    ymd_regex = r"_(\d{4})_(\d{2})_(\d{2})"
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):

            year, month, day = [
                int(number) for number in re.findall(ymd_regex, file)[0]
            ]
            dates.append(date(year, month, day))

    return dates


def get_day_of_the_year(dates: list[date]) -> list[int]:
    """Get the day of the year"""
    days = [d.timetuple().tm_yday for d in dates]

    return days


def select_datavars(dataset: xr.Dataset) -> xr.Dataset:
    """
    Select only relevant variables (for Brightness temperature)
    Learn more: https://www.remss.com/missions/windsat/
    """

    selected_dvars = [
        "longitude",
        "latitude",
        "node",  # node of swath, [ascending, descending]
        # "look", # look direction. we will select only look = 0 (forward)
        "frequency_vpol",  # center frequency of V-pol channel in each band
        "frequency_hpol",  # center frequency of H-pol channel in each band
        "eia_nominal",  # nominal Earth indidence angle of each band
        "time",  # Time of observation (lat, lon) seconds since 01 JAN 2000 00Z
        "eaa",  # boresight Earth azimuth angle. range: [0o, 360o].
        "eia",  #  boresight Earth incidence angle. range: [0o, 90o]
        "tbtoa",  # Brightness temperature
        "quality_flag",  # 32-bit quality control flag
        # "sss_HYCOM", # HYCOM sea surface salinity
        # "surtep_REY", # NOAA (Reynolds) V2 OI sea surface temperature
        # # Land fractions
        # "fland_06", # for 6GHz
        # "fland_10", # For 10 GHz
        # # Windsat V8 products
        # "surtep_WSAT", # skin temperature
        # "colvap_WSAT", # atmosphere_mass_content_of_water_vapor
        # "colcld_WSAT", # atmosphere_mass_content_of_cloud_liquid_water
        # "winspd_WSAT", # sea surface wind speed
        # "rain_WSAT", # surface rain rate
        # # Cross-Calibrated Multi-Platform
        # "winspd_CCMP", # Wind speed
        # "windir_CCMP", # Cross-Calibrated Multi-Platform Wind direction
        # # ERA 5 products
        # "surtep_ERA5", # skin temperature
        # "airtep_ERA5", # Air temperature at 2m above surface
        # "colvap_ERA5", # Columnar liquid cloud water
        # "colcld_ERA5", # atmosphere_mass_content_of_cloud_liquid_water
        # "winspd_ERA5", # 10-m NS wind speed
        # "windir_ERA5", # Wind direction
        # "surtep_CMC", # CMC Sea surface temperature
        # "rain_IMERG", # IMERG V6 surface rain rate
        # # RSS 2022 absorption model
        # "tran", # Total atmospheric transmittance computed from ERA atmospheric profiles and WSAT columnar vapor and cloud water
        # "tbdw", # Atmospheric downwelling brightness temperature computed from ERA atmospheric profiles and WSAT columnar vapor and cloud water
    ]

    return dataset[selected_dvars]


def select_dims(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove unused frequencies and polarizations

    Frequencies:
    0 -- 6.8 GHz
    1 -- 10.7 GHz
    2 -- 18.7 GHz (Ku)
    3 -- 23.8 GHz
    4 -- 37.0 GHz (Ka)

    Polarizations:
    0 -- V
    1 -- H
    (except for 6.8 and 23.8 GHz):
    2 -- P (+45º)
    3 -- M (-45º)
    4 -- L (Circular Left)
    5 -- R (Circular Right)
    """

    # Select dimensions
    dataset = dataset.sel(
        indexers={
            "polarization": [0, 1],  # [ V, H ]
            "frequency_band": [2, 4],  # [ 18.7 GHz (Ku) , 37.0 GHz (Ka) ]
            "look_direction": 0,  # Forward
        }
    )
    return dataset


def transform_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Other transformations
    """
    # Roll lattitude grid so we have -180, 180 range
    dataset = dataset.roll(shifts={"longitude_grid": 4 * 180})

    # Extract latitude and longitude grid dimensions
    dataset = dataset.assign_coords(lat=dataset.latitude, lon=dataset.longitude)

    return dataset


def preporcess_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Wrapper"""
    dataset = select_datavars(dataset)
    dataset = select_dims(dataset)
    dataset = transform_dataset(dataset)
    return dataset


_preprocess_dataset = partial(preporcess_dataset)


def windsat_datacube(folder_path: str) -> xr.Dataset:
    """
    Wrapper for creating a dataset with the combined data inside a folder
    param folder_path: must contain the files in .nc format
    """

    dates = recover_dates(folder_path)
    day_numbers = get_day_of_the_year(dates)

    ds = xr.open_mfdataset(
        paths=folder_path + "\\*.nc",
        preprocess=_preprocess_dataset,
        decode_times=False,  # "time" is a datavar (time of observation for each pixel)
        concat_dim="day_number",
        combine="nested",
    )

    # Add a day_number coordinate
    ds["day_number"] = day_numbers
    ds["day_number"].attrs = {f"Description": "Int, day of the year {dates[0].year}"}

    return ds
