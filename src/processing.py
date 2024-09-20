import zipfile
import io
import xarray as xr
import regionmask
import re
import os
import numpy as np
from datetime import datetime, date
from functools import partial
from tqdm import tqdm
import warnings
from scipy.ndimage import distance_transform_edt
from typing import Any


# TODO: use this instead of hardcodin the values.
def holmes(brightness_temp: xr.Dataset) -> xr.Dataset:
    """
    Calculates the surface temperature from the brightness
    temperature using Holmes formula.
    Only valid for Tb (V) and avobe 259.8 K (Brightness Temperature)
    """
    a = 1.11
    b = -15.2

    return a * brightness_temp + b


# NOTE: used only in the dataset for SWF
def apply_scaling(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply scaling factors to data variables in an
    Xarray Dataset based on the "SCALE FACTOR" attribute.

    :param xarray_dataset: The input Xarray Dataset.
    :return: A new Xarray Dataset with scaled variables.
    """

    for dvar in ds.data_vars:
        if "SCALE FACTOR" in ds[dvar].attrs.keys():
            scale_factor = ds[dvar].attrs["SCALE FACTOR"]
            if scale_factor != 1:
                ds[dvar] *= scale_factor
                ds[dvar].attrs["SCALE FACTOR"] = 1  # Update the attribute

    return ds


# TODO: use multi-folder loading from xarray instead.
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
                        # We eliminate the two first days,
                        # since they are not valid.
                        continue

                    # Determine the class from the file name (EQMA or EQMD)
                    file_class = (
                        "Ascending" if "EQMA" in file_name else "Descending"
                    )
                    # Add the 'time' and 'class' coordinates to the dataset
                    h5_xarray["time"] = date
                    h5_xarray["Pass"] = file_class

                    # Append the dataset to the list
                    xarray_datasets.append(h5_xarray)

        # Concatenate the list of datasets
        xarray_dataset = xr.concat(xarray_datasets, dim="time")

    # Now, xarray_dataset contains all data with 'time' and 'class'
    # coordinates to distinguish between EQMA and EQMD
    xarray_dataset = xarray_dataset.set_coords("Pass")

    return xarray_dataset


# Currently unused, but a good idea I gess
def extract_timeseries(xarray_dataset, lat, lon) -> xr.DataArray:
    """
    Extract the timeseries from a pixel in the map for all instances of time

    :param xarray_dataset: Dataset with at least lat lon and time coordinates
    :param lat: float
    :param lon: float
    :return: np.array 1D
    """
    warnings.warn(
        "DEPRECATED: Please use xarray.sel(lat=lat, lon=lon) instead."
    )
    timeseries = np.zeros(shape=(len(xarray_dataset.time),))

    for i in range(len(xarray_dataset.time)):
        timeseries[i] = xarray_dataset[i, lat, lon]

    return np.array(timeseries)


def recover_dates(
    folder_path: str, ymd_regex: str = r"_(\d{4})_(\d{2})_(\d{2})"
) -> list[date]:
    """
    Returns a list of dates parsing the file names within the folder path
    using the given year-month-day regex pattern.

    param folder_path:
    param ymd_regex: must return year, month and day using capturing groups.
    Default works for Windsat datafiles:
    "RSS_WINDSAT_DAILY_TBTOA_MAPS_2017_01_01.nc"
    """

    dates = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):

            year, month, day = [
                int(number) for number in re.findall(ymd_regex, file)[0]
            ]
            dates.append(date(year, month, day))

    return dates


def get_day_of_the_year(dates: list[date]) -> list[int]:
    """Return the day of the year from a list of datetime objects"""
    return [d.timetuple().tm_yday for d in dates]


def select_datavars(dataset: xr.Dataset) -> xr.Dataset:
    """
    Select only relevant variables (for Brightness temperature)
    Learn more: https://www.remss.com/missions/windsat/
    """

    selected_dvars = [
        # Datavars coding for Dimensions
        "longitude",
        "latitude",
        "node",  # node of swath, [ascending, descending]
        # "look", # look direction. we will select only look = 0 (forward)
        # Actual data
        "frequency_vpol",  # center frequency of V-pol channel in each band
        "frequency_hpol",  # center frequency of H-pol channel in each band
        "eia_nominal",  # nominal Earth indidence angle of each band
        "time",  # Time of observation (lat, lon) seconds since 01 JAN 2000 00Z
        "eaa",  # boresight Earth azimuth angle. range: [0o, 360o].
        "eia",  # boresight Earth incidence angle. range: [0o, 90o]
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
        "surtep_ERA5",  # skin temperature
        "airtep_ERA5",  # Air temperature at 2m above surface
        # "colvap_ERA5", # Columnar liquid cloud water
        # "colcld_ERA5", # atmosphere_mass_content_of_cloud_liquid_water
        # "winspd_ERA5", # 10-m NS wind speed
        # "windir_ERA5", # Wind direction
        # "surtep_CMC", # CMC Sea surface temperature
        # "rain_IMERG", # IMERG V6 surface rain rate
        # # RSS 2022 absorption model
        # "tran", # Total atmospheric transmittance computed from ERA
        # atmospheric profiles and WSAT columnar vapor and cloud water
        # "tbdw", # Atmospheric downwelling brightness temperature computed
        # from ERA atmospheric profiles and WSAT columnar vapor and cloud water
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
    (except for 6.8 and 23.8 GHz we also have):
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
            # "look_direction": 1,  # After look
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
    dataset = dataset.assign_coords(
        lat=dataset.latitude, lon=dataset.longitude
    )

    return dataset


def preporcess_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Wrapper"""
    dataset = select_datavars(dataset)
    dataset = select_dims(dataset)
    dataset = transform_dataset(dataset)
    return dataset


# Partial function definition
_preprocess_dataset = partial(preporcess_dataset)


def impute_look_data(ds: xr.Dataset, add_look_flag: bool = True) -> xr.Dataset:
    """
    Linear regression Tb(look = 0) = a·Tb(look=1) + b
    with pre-computed coefficients with a sample of the full dataset

    ds: Dataset with look_direction dimension
    param add_flag: Default True. Add an additional mask to flag imputed data.

    returns: ds without look_direction, tbtoa missing data in look = 0 is
    imputed with LinReg(tbtoa_look=1).
    """

    # Pre-computed coefficients for the linear regression
    # TODO: Script to fit this coefficients from data,
    # save them into a file and use them here.
    coeffs = np.array([
        # H Pol
        [
            [(0.9823, 5.092), (0.98584, 3.756)],  # 37GHz
            [(0.98662, 3.916), (0.98335, 4.519)]   # 19GHz
            # Ascending          Descending
        ],
        # V Pol
        [
            [(0.99922, -0.01), (1.00056, -0.589)],  # 37GHz
            [(1.00278, -1.381), (1.00578, -1.68)]  # 19GHz
            # Ascending          Descending
        ]
    ])

    if "look_direction" not in ds.dims:
        Warning(
            "Provided dataset has no look_direction, dataset returned as is."
        )
        return ds

    fore = ds.sel(look_direction=0)
    aft = ds.sel(look_direction=1)

    if add_look_flag:
        # Flag all data that can be imputed.
        can_impute = aft.where(np.isnan(fore.tbtoa))
        ds["imputed_flag"] = ~can_impute.tbtoa.isnull()

    # Apply the coefficients to the aft look
    slopes = xr.DataArray(
        coeffs[..., 0],
        dims=["polarization", "frequency_band", "swath_sector"]
    )
    intercepts = xr.DataArray(
        coeffs[..., 1],
        dims=["polarization", "frequency_band", "swath_sector"]
    )

    imputed_tbtoa = aft.tbtoa * slopes + intercepts

    # Fill missing values in the fore data with the imputed values
    filled_tbtoa = fore.tbtoa.fillna(imputed_tbtoa)

    # Replace the original 'tbtoa' DataArray with the filled data
    ds['tbtoa'] = filled_tbtoa

    return ds


def windsat_datacube(folder_path: str) -> xr.Dataset:
    """
    Wrapper for creating a dataset with the combined data inside a folder
    param folder_path: must contain the files in .nc format

    Limited to general transformations
    """

    dates = recover_dates(folder_path)
    day_numbers = get_day_of_the_year(dates)

    ds = xr.open_mfdataset(
        paths=os.path.join(folder_path + "/*.nc"),
        preprocess=_preprocess_dataset,
        decode_times=False,
        # "time" is a datavar (time of observation for each pixel)
        concat_dim="day_number",
        combine="nested",
    )

    # Add a day_number coordinate
    ds["day_number"] = day_numbers
    ds["day_number"].attrs = {
        "Description": f"Int, day of the year {dates[0].year}"
    }

    return ds


def create_landmask(
        lat: np.array,
        lon: np.array,
        c_dist: float = None) -> xr.DataArray:
    """
    Return a landmask without pixels that are c_dist
    or closer to a coast pixel.
    Default None: Do not remove coastline pixels.

    lon: 1D array with all the longitude values in the array
    lat: 1D "                "  latitude  "                 "

    returns a 2D DataArray (lat x lon) with the 0 flag for land/included
    NaN for ocean/excluded.
    """

    land = regionmask.defined_regions.natural_earth_v5_1_2.land_10
    landmask = land.mask(lon_or_obj=lon, lat=lat)

    if c_dist:
        # 0 (land) and nan (ocean) values, add 1 to the array so land == 1
        aux_landmask = landmask + 1

        # Then fill the nan values with 0
        aux_landmask = aux_landmask.fillna(0)

        # Distance from each point to the closest ocean pixel
        # (from all values > 0 to all values == 0)
        coastline_dist = distance_transform_edt(aux_landmask.values)

        # New mask, only the pixels c_dist away from the closest ocean pixel.
        coasline_mask = coastline_dist <= c_dist

        # Remove the coastline from the original landmask
        landmask = landmask.where(~coasline_mask)

    return landmask


def model_preprocess(
        ds: xr.Dataset,
        swath_sector: int = 0,
        look: Any = "impute",
        add_look_flag: bool = True
        ) -> xr.Dataset:
    """
        Pre-process windsat dataset to:
         -add landmask
         -selecct a swath
         -remove ERA5 skin temperature over 2ºC
         -impute missing look data
         -transform polarizations and frequency slices of tbtoa into
        separated dvars.

        param swath_sector: 0 for Ascending pass, 1 for Descending pass
        param look: int or Any: select Fore look (0), Aft look (1),
        Default: "impute" missing
        Fore data with linear regression models and Aft data.
        param add_look_flag: add a boolean dvar with wheather
        or not the data was imputed.

    """
    # Preprocess and select the dataset
    landmask = create_landmask(lon=ds.lon.values, lat=ds.lat.values)
    ds["landmask"] = (("latitude_grid", "longitude_grid"), landmask.values)

    # Filter the dataset for land
    ds = ds.where(ds.landmask == 0)

    # select data only where era5 surtep is avobe 2ºC
    ds = ds.where(ds.surtep_ERA5 > (273.15 + 2))

    # Look data handling:
    if isinstance(look, int):
        ds = ds.sel(look_direction=look)

    elif look == "impute":
        ds = impute_look_data(ds, add_look_flag)

    # Select desired swaht
    ds = ds.sel(swath_sector=swath_sector)

    # Select only desired variables
    variables = ["tbtoa", "surtep_ERA5"]
    ds = ds[variables]

    # Split tbtoa and time into polarization and frequency
    ds["tbtoa_18Ghz_V"] = ds.tbtoa.sel(polarization=0, frequency_band=0)
    ds["tbtoa_18Ghz_H"] = ds.tbtoa.sel(polarization=1, frequency_band=0)
    ds["tbtoa_37Ghz_V"] = ds.tbtoa.sel(polarization=0, frequency_band=1)
    ds["tbtoa_37Ghz_H"] = ds.tbtoa.sel(polarization=1, frequency_band=1)

    # Drop the original dvar
    ds = ds.drop_vars(names=["tbtoa"])

    # Lat and lon should be dvars instead
    ds = ds.reset_coords(names=["lat", "lon"])

    # Add longitude_grid and latitude_grid as indeces
    ds = ds.assign_coords(latitude_grid=range(720), longitude_grid=range(1440))
    ds = ds.set_index(
        latitude_grid='latitude_grid',
        longitude_grid='longitude_grid'
    )

    return ds


def recover_months(filenames: list[str]) -> list[int]:
    """
    Read the name files and recover the month using regex.
    "" ssmi_mean_emis_climato_MM_cov_interpol_M2.nc ""
    """
    regex_pattern = r"ssmi_mean_emis_climato_(\d{2})_cov_interpol_M2.nc"

    months = [
        int(re.findall(regex_pattern, fn)[0]) for fn in filenames
    ]
    return months


def telsem_preprocess(telsem_ds: xr.Dataset) -> xr.Dataset:
    """
    Select datavars, roll longitude, add landmask, keep land, reset coords
    """
    d_vars = [
        "Emis19V",
        "Emis19H",
        "Emis37V",
        "Emis37H",
    ]

    telsem_ds = telsem_ds[d_vars]

    # Roll the longitude to align the data
    telsem_ds = telsem_ds.roll(
        {
            "longitude_grid": 4 * 180
        }
    )

    landmask = create_landmask(
        lat=telsem_ds.lat.values,
        lon=telsem_ds.lon.values
    )
    telsem_ds["landmask"] = (
        ("latitude_grid", "longitude_grid"), landmask.values
    )

    telsem_ds = telsem_ds.where(telsem_ds.landmask == 0)
    telsem_ds = telsem_ds.drop_vars("landmask")
    telsem_ds = telsem_ds.reset_coords()

    return telsem_ds


_telsem_preprocess = partial(telsem_preprocess)


def telsem_datacube(folder_path: str) -> xr.Dataset:
    """
    Same idea as windsat_datacube, preprocess the files in the folder,
    add the month as a dimention, roll the longitude grid, etc.
    """
    filenames = [
        fn for fn in os.listdir(folder_path)
        if fn.endswith(".nc")
    ]
    months = recover_months(filenames)

    telsem_ds = xr.open_mfdataset(
        paths=[
            os.path.join(folder_path, fn) for fn in filenames
        ],
        preprocess=_telsem_preprocess,
        concat_dim="month",
        combine="nested"
    )

    telsem_ds["month"] = months
    telsem_ds["month"].attrs = {
        "Description": "Month of the year"
    }

    return telsem_ds


def doy2month_mapping() -> list[int]:
    """
    Create the day to month mapping list
        Given the Day of the year 'DoY', return its
        corresponding month of the year by index.

        day_mapping[31] = 1 # JAN
        day_mapping[60] = 2 # FEB
        [...]
        day_mapping[366] = 12 # Dec
    """
    days_in_month = [
        31,  # JAN
        29,  # FEB (on leep years)
        31,  # MAR
        30,  # APR
        31,  # MAY
        30,  # JUN
        31,  # JUL
        31,  # AUG
        30,  # SEP
        31,  # OCT
        30,  # NOV
        31,  # DEC
    ]
    day_mapping = []
    for i, n in enumerate(days_in_month):
        add = [i+1] * n
        day_mapping.extend(add)

    return day_mapping
