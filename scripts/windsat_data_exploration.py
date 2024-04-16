""" 
    Read and load a dataset with all .nc files from a folder (WINDSAT DATA 2017)

    Preprocess and agregate data to return measurement counts, averages and other

    Save the results into separated .nc files, plot and save .png images.
"""

# Imports
import matplotlib.pyplot as plt
import xarray as xr
import os
import re
from datetime import date
from functools import partial

from argparse import ArgumentParser

params = ArgumentParser()

params.add_argument(
    "--source-folder", default="./", help="Data folder path, contatining all .nc files"
)

params.add_argument(
    "--save-folder", default="./", help="Aggregated data will be saved here"
)

params.add_argument(
    "--plots",
    default=False,
    help="Python Boolean [True|False], whether or not to compute and save plots",
)


# TODO: For now this is OK, but I need to get the repo into the ERC server and
# Import custom functions instead of defining them here.
# Local functions:
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
        paths=os.path.join(folder_path, "*.nc"),
        preprocess=_preprocess_dataset,
        decode_times=False,  # "time" is a datavar (time of observation for each pixel)
        concat_dim="day_number",
        combine="nested",
    )

    # Add a day_number coordinate
    ds["day_number"] = day_numbers
    ds["day_number"].attrs = {f"Description": "Int, day of the year {dates[0].year}"}

    return ds


# Custom plot function
def dimensional_plot(
    ds: xr.DataArray, save_path: str = None, cbar_label: str = None
) -> None:
    """
    Given a data_array, plot each combination of:
        polarization : [0,1]
        frequency: [0,1]
        swath_sector : [0,1]

    param save_path: if not None, atempt to save the plot with the given path
    param cbar_label: if not None, swap the default xarray.plot label for it
    """

    dimension_dict = {
        "polarization": {0: "V", 1: "H"},
        "frequency": {0: "18.7GHz", 1: "37.0GHz"},
        "sector": {0: "Asc", 1: "Des"},
    }

    fig, axs = plt.subplots(4, 2, figsize=(8, 12))

    plot_number = 0
    for sector in range(0, 2, 1):
        for freq in range(0, 2, 1):
            for pol in range(0, 2, 1):
                nrow = plot_number // 2
                ncol = plot_number % 2
                ax = axs[nrow, ncol]

                # Plot data
                plot = ds.sel(
                    polarization=pol, frequency_band=freq, swath_sector=sector
                ).plot(ax=ax)

                # TODO: Add coastline
                """ 
                Data is in 1/4º grid, cartopy uses latitude and longitude.
                    - Change the grid
                    OR
                    - Change the coastline feature somehow.
                """
                # ax.coastlines(resolution="110m", color = "white", linewidth=1)

                ax.set_title(
                    f"Freq: {dimension_dict['frequency'][freq]}, "
                    f"Pol: {dimension_dict['polarization'][pol]}, "
                    f"Swath: {dimension_dict['sector'][sector]}"
                )
                if cbar_label:
                    colorbar = plot.colorbar
                    colorbar.set_label(cbar_label)

                plot_number += 1

    fig.tight_layout()
    if save_path:
        try:
            fig.savefig(save_path)
        except Exception as e:
            print(f"Unable to save plot: {e}")

    return


if __name__ == "__main__":

    args = params.parse_args()
    source_path = args.source_folder
    save_path = args.save_folder
    plots = args.plots

    ds = windsat_datacube(source_path)

    # Get the number of measurements per pixel.
    tbtoa_count = ds.tbtoa.count(dim="day_number")
    tbtoa_count.attrs = {
        "Description": "Total count of measurements in the whole datset",
    }
    save_into = os.path.join(save_path, "tbtoa_count.nc")
    tbtoa_count.to_netcdf(save_into)
    print(f"TBToA count saved in {save_into}")

    # Get the average TBToA per pixel.
    tbtoa_mean = ds.tbtoa.mean(dim="day_number")
    tbtoa_mean.attrs = {
        "Description": "Yearly average TbToA ",
    }
    save_into = os.path.join(save_path, "tbtoa_mean.nc")
    tbtoa_mean.to_netcdf(save_into)
    print(f"TBToA count saved in {save_into}")

    if plots:
        # Generate and save the plots
        dimensional_plot(
            tbtoa_count,
            save_path=os.path.join(save_path, "TBToA_count.png"),
            cbar_label="TBToA Count",
        )
        dimensional_plot(
            tbtoa_mean,
            save_path=os.path.join(save_path, "TBToA_mean.png"),
            cbar_label="Mean TBToA",
        )

    print("DONE")
