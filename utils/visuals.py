import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def show_map(image: np.array, title=None, save_path=None):
    """
    This function will show a map of the image
    :param save_path:
    :param title:
    :param image:
    """
    # Fix the orientation issue by using coordinates
    image = np.flip(image, axis=0)

    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap="jet")
    cbar = plt.colorbar()
    cbar.set_label("Average Surface Temperature (K)")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def histogram_matrix(h5file, save_path=None):
    """
    This function will show an image with the histograms of the different fields of the hdf5 file
    :param save_path:
    :param h5file:
    :return:
    """
    fields = h5file.keys()
    field_number = len(fields) - 2  # Remove lat and lon
    row_number = field_number // 2
    col_number = 2

    fig, axs = plt.subplots(row_number, col_number, figsize=(10, 20))
    for i, field in enumerate(fields):
        if field != "lat" & field != "lon":
            axs[i // col_number, i % col_number].hist(
                h5file[field][:].flatten(), bins=100
            )
            axs[i // col_number, i % col_number].set_title(field)

        if save_path is not None:
            plt.savefig(f"{save_path}.png")
    plt.show()


def column_plot(xarray_dataset, save_path=None):
    """
    Creates the subplots for each of the passes in the whole dataset
    """
    ncols = 2  # Separate ascending and descending passes into two columns
    nrows = (
        len(xarray_dataset.time) // 2 + 1
        if len(xarray_dataset.time) % 2 == 1
        else len(xarray_dataset.time) // 2
    )
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 12))

    for i in range(len(xarray_dataset.time)):
        row = i // ncols
        col = i % ncols

        xarray_dataset["Holmes_LST"][i, :, :].plot(ax=axs[row, col], cmap="jet")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

# Custom function for windsat data
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
