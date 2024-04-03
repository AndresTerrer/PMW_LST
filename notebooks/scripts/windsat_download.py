""" 
    Automatically download daily aggregated data from Windsat
    https://images.remss.com/~RSS-TB/intercalibration/windsat_TB_maps_daily_025deg/

    Data is available from 2017-01-01 to 2019-12-31
    
    Contact :
    tegoan@uv.es
"""
import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from argparse import ArgumentParser
import logging

logger = logging.getLogger("windsatDonload")

params = ArgumentParser(
    description = "Download a set of files from windsat daily data",
)

params.add_argument(
    "--save_into",
    default= "./",
    help= "Target directory to save the files into"
)

params.add_argument(
    "--start_date",
    default="2017-01-01",
    help="Date in YYYY-MM-DD format"
)

params.add_argument(
    "--end_date",
    default="2017-01-31",
    help="Date in YYYY-MM-DD format"
)


async def download_file(url, save_path):
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)
        ) as session:
            async with session.get(url) as response:
                with open(save_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            logger.info(f"Downloaded {save_path}")
                            break
                        f.write(chunk)
    except aiohttp.ClientError as e:
        logger.info(f"Error downloading {url}: {e}")


async def concurrent_download(dates, save_dir):
    tasks = []
    base_url = "https://images.remss.com/~RSS-TB/intercalibration/windsat_TB_maps_daily_025deg/RSS_WINDSAT_DAILY_TBTOA_MAPS"

    for date in dates:
        year = str(date.year).zfill(4)
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)

        url = f"{base_url}_{year}_{month}_{day}.nc"
        save_file = url.split("/")[-1]
        save_path = os.path.join(save_dir, save_file)

        if not os.path.exists(save_path):
            logger.info(f"Queuing download for {save_file}")
            tasks.append(download_file(url, save_path))

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    args = params.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    save_dir = args.save_into

    dates = [
    start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]

    asyncio.run(concurrent_download(dates, save_dir))
    logger.info("DONE")