""" 
    Automatically download hourly ERA5-land data between two dates.
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview

    Data is available from 1950 to "present" (atm. April 2024)

    A profile is needed in order to call the api
    https://pypi.org/project/cdsapi/

    Contact :
    tegoan@uv.es
"""

import cdsapi
import os
from datetime import timedelta, datetime
from argparse import ArgumentParser

import logging

logger = logging.getLogger("ERA5_Land_donwload")
logger.setLevel(logging.INFO)  # Set the logging level to INFO

params = ArgumentParser(
    description="Download ERA5_land hourly data between two dates",
)

params.add_argument(
    "--save-into",
    default="C:/Users/andre/Desktop/VS_Code/PMW_LST/data/raw/daily_Windsat/",
    help="Target directory to save the files into",
)

params.add_argument(
    "--start-date", default="2017-01-01", help="Date in YYYY-MM-DD format"
)

params.add_argument(
    "--end-date", default="2017-03-02", help="Date in YYYY-MM-DD format"
)


def get_base_request() -> dict:
    """ 
    For the originaly intended dataset and variables.
    """
    dataset_name = "reanalysis-era5-land"
    all_hours = [f"{hour:02d}:00" for hour in range(24)]

    base_request = {
        "variable":
        [
            '2m_temperature', 'skin_temperature', 'soil_temperature_level_1',
        ],
        "year": None,
        "month": None,
        "day": None,
        "time": all_hours,
        "format": "netcdf.zip",
    }

    
    return base_request, dataset_name


def single_download(cds:cdsapi.Client, date:datetime, folder_path:str) -> None:
    request, dataset_name = get_base_request()

    # Update the  base request using the date
    request["year"] = str(date.year).zfill(4)
    request["month"] = str(date.month).zfill(2)
    request["day"] = str(date.day).zfill(2)

    # Name the target file acordingly:
    filename = f"{request['year']}_{request['month']}_{request['day']}.zip"
    target = os.path.join(folder_path, filename)

    # Process only unexisting, full downloads
    if  not os.path.exists(target):
        try:
            cds.retrieve(
                name=dataset_name,
                request=request,
                target=target,
            )
        except Exception as e:
            os.remove(target)
            print(f"Removed compromised file {target}.")
            print(e)

    else:
        print(f"Data file {target} allready exists, donwload skipped.")


def download_many(dates:list[datetime], save_dir:str) -> None:

    cds = cdsapi.Client()

    for i,date in enumerate(dates):
        print(f"Downloading atempt {i}/{len(dates)}.")
        single_download(cds, date, save_dir)
        print("NEXT")

    print("DONE")


if __name__ == "__main__":

    args = params.parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    save_dir = args.save_into

    dates = [
        start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]

    download_many(dates, save_dir)


    

