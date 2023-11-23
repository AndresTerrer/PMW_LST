import matplotlib.pyplot as plt
import numpy as np


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
