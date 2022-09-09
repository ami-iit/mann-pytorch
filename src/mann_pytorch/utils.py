import os
import glob
import json
import os.path
import numpy as np


def get_latest_model_path(models_path: str) -> str:
    """Retrieve the path of the latest saved model.

    Args:
        models_path (str): The folder in which the learned models are stored

    Returns:
        latest_model (str): The path of the latest saved model
    """

    list_of_files = glob.glob(models_path + '*')
    latest_model = max(list_of_files, key=os.path.getctime)
    print("Latest retrieved model:", latest_model)

    return latest_model


def create_path(path: str) -> None:
    """Create a path if it does not exist.

    Args:
        path (str): The path to be created
    """

    if not os.path.exists(path):
        os.makedirs(path)


def normalize(X: np.array, axis: int) -> np.array:
    """Normalize a matrix along the given axis.

    Args:
        X (np.array): The matrix to be normalized
        axis (int): The axis along which the matrix has to be normalized

    Returns:
        X_norm (np.array): The normalized matrix
    """

    # Compute mean and std
    Xmean = X.mean(axis=axis)
    Xstd = X.std(axis=axis)

    # Avoid division by zero
    for elem in range(Xstd.size):
        if Xstd[elem] == 0:
            Xstd[elem] = 1

    # Normalize
    X_norm = (X - Xmean) / Xstd

    return X_norm


def denormalize(X: np.array, Xmean: np.array, Xstd: np.array) -> np.array:
    """Denormalize a matrix, given its mean and std.

    Args:
        X (np.array): The matrix to be denormalized
        Xmean (np.array): The vector of means to be used for normalization
        Xstd (np.array): The vector of standard deviations to be used for normalization

    Returns:
        X_denorm (np.array): The denormalized matrix
    """

    # Denormalize
    X_denorm = X * Xstd + Xmean

    return X_denorm


def store_in_file(data: list, filename: str) -> None:
    """Store data in file as json.

    Args:
        data (list): The data to be stored
        filename (str): The storage filename
   """

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def read_from_file(filename: str) -> np.array:
    """Read data as json from file.

    Args:
        filename (str): The name of the file containing the data to be retrieved

    Returns:
        data (np.array): The retrieved data
   """

    with open(filename, 'r') as openfile:
        data = np.array(json.load(openfile))

    return data

