""" 
Reproject the whole folder containign GeoTIF files (LPDR2)

Needs to be run using conda (gdal api). python >= 3.9
$ conda install -c conda-forge gdal

other dependencies: 
numpy
"""

import os
import numpy as np
import re

from osgeo import gdal, osr
from argparse import ArgumentParser

params = ArgumentParser()
params.add_argument(
    "--source_folder", default="./data/", help= "Folder with LPDR's GeoTIF."
)

params.add_argument(
    "--output_folder", default = None, help = "Folder to save a copy of the reprojected files. Default: None will overwrite the files instead."
)


def reproject_file (file_path: str, output_folder: str = None) -> bool:
    """  
        Read the input geotiff in EASE v1
        Reproject + resample into the EASE v2
        create latitude an longitude bands for convenience.
        (bands added as second to last, and last band respectively)
        
        param output_folder: name of the new file to save reprojected data. 
            If None, data will be ovewritten in file_path.        

        Returns bool: whether or not the file was succesfully reprojected
    """
    dataset = gdal.Open(file_path)        

    # Define the geotransform for the output dataset
    target_geotransform = (-17367530.44, 25025.26, 0.0, 7307375.92, 0.0, -25025.26)

    output_width = 1388
    output_height = 584

    # Define src and geotransform from the input:
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(3410)

    # Desired projection 
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(6933)

    # Inverse transformation to geographic coordinates (to add lon & lat)
    geo_srs = osr.SpatialReference()
    geo_srs.ImportFromEPSG(4326)  # WGS 84
    inverse_transform = osr.CoordinateTransformation(target_srs, geo_srs)

    # Declare the output file and driver:
    driver = gdal.GetDriverByName("GTiff")

    if output_folder is None:
        output_file = os.path.join(os.path.dirname(file_path), "temp.tif")

    else:
        output_file = os.path.join(output_folder, os.path.basename(file_path))

    # Create the new dataset.
    output_dataset = driver.Create(output_file, output_width, output_height, dataset.RasterCount + 2, gdal.GDT_Float32)
    output_dataset.SetProjection(target_srs.ExportToWkt())
    output_dataset.SetGeoTransform(target_geotransform)

    # Reproject and resample using gdal.Warp()
    gdal.Warp(
        output_dataset,
        dataset,
        dstSRS=target_srs.ExportToWkt(),
        width=output_width,
        height=output_height,
        resampleAlg=gdal.GRA_Bilinear, # GRA_Bilinear TODO: bilinear warp does not work with Nodata params, there will be values between -999 and the valid range
        srcNodata = -999.0,
        dstNodata = -999.0
    )

    # Create bands for latitude and longitude
    lat_band = output_dataset.GetRasterBand(dataset.RasterCount + 1)
    lon_band = output_dataset.GetRasterBand(dataset.RasterCount + 2)

    # Create arrays for latitude and longitude
    lat_array = np.zeros((output_height, output_width), dtype=np.float32)
    lon_array = np.zeros((output_height, output_width), dtype=np.float32)

    # Calculate latitude and longitude for each pixel
    for y in range(output_height):
        for x in range(output_width):
            pixel_x = target_geotransform[0] + x * target_geotransform[1] + y * target_geotransform[2]
            pixel_y = target_geotransform[3] + x * target_geotransform[4] + y * target_geotransform[5]
            lon, lat, _ = inverse_transform.TransformPoint(pixel_x, pixel_y)
            lat_array[y, x] = lat
            lon_array[y, x] = lon

    # Write the latitude and longitude arrays
    lat_band.WriteArray(lat_array)
    lon_band.WriteArray(lon_array)

    # Set descriptions
    lat_band.SetDescription('Latitude')
    lon_band.SetDescription('Longitude')

    # Close the files
    dataset = None
    output_dataset = None

    if output_folder is None:
        # Delete original file, rename temp file.
        os.remove(file_path)
        os.rename(output_file, file_path)

    return True


if __name__ == "__main__":  

    args = params.parse_args()

    source_folder = args.source_folder
    output_folder = args.output_folder

    print("START")

    # NOTE: Do not reproject Quality Flag files for now, those files end in '\d{3}QA.tif'
    # Select only ascending and descending passes:
    regex = r"\d{7}[AD].tif"

    for file_name in os.listdir(source_folder):
        if re.findall(regex, file_name):

            file_path = os.path.join(source_folder, file_name)
            print(f"Reprojecting {file_path}")

            outcome = reproject_file(file_path, output_folder = output_folder)

            if outcome:
                print("Reprojected")
                if output_folder is not None:
                    print(f"New file saved {os.path.join(output_folder, file_name)}")
            else:
                print(f"File {file_name} is already reprojected")
    
    print("DONE")