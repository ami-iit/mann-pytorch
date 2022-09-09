import json
import numpy as np
from typing import List
from datetime import datetime
from torch.utils.data import Dataset
from mann_pytorch.utils import create_path, normalize, store_in_file


class CustomDataset(Dataset):
    """Class for a custom PyTorch Dataset."""

    def __init__(self, X: np.array, Y: np.array):
        """Constructor of the custom dataset.

        Args:
            X (np.array): The vector of inputs
            Y (np.array): The vector of outputs
        """

        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        """Getter of the dataset length.

        Returns:
            dataset_length (np.array): The number of (X,Y) instances included in the dataset
        """

        dataset_length = len(self.X)
        return dataset_length

    def __getitem__(self, idx) -> (np.array, np.array):
        """Getter of an element of the dataset.

        Args:
            idx (int): The index of the desired element in the dataset

        Returns:
            x_idx (np.array): The input vector at the specified index in the dataset
            y_idx (np.array): The output vector at the specified index in the dataset
        """

        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        return x_idx, y_idx

    def get_input_size(self) -> int:
        """Getter of the input size.

        Returns:
            input_size (int): The size of the input vectors in the dataset
        """

        input_size = len(self.X[0])
        return input_size

    def get_output_size(self) -> int:
        """Getter of the output size.

        Returns:
            output_size (int): The size of the output vectors in the dataset
        """

        output_size = len(self.Y[0])
        return output_size


class DataHandler:
    """Class for processing the data in order to get the training and testing sets."""

    def __init__(self, input_paths: List, output_paths: List, storage_folder: str, training: bool, training_set_percentage: int):
        """DataHandler constructor.

        Args:
            input_paths (List): The list of filenames containing all the inputs to be stacked
            output_paths (List): The list of filenames containing all the outputs to be stacked
            storage_folder (str): The initial name of the local folder to be used for storage
            training (bool): Flag to indicate whether the DataHandler will be used also for training (True) or for testing only (False)
            training_set_percentage (int): The percentage of the data to be used for training
        """

        # Store the paths to the inputs and outputs of the network
        self.input_paths = input_paths
        self.output_paths = output_paths

        # Store the path for the training-related data
        self.savepath = self.define_savepath(storage_folder)

        # Retrieve the training and testing data
        self.training_data, self.testing_data = self.retrieve_training_and_testing_data(training, training_set_percentage)

    @staticmethod
    def define_savepath(storage_folder: str) -> str:
        """Update the storage folder name by adding the current timing so to have different savepaths for each training.

        Args:
            storage_folder (str): The initial name of the local folder to be used for storage

        Returns:
            savepath (str): The updated name of the local folder to be used for storage, including the current timing
        """

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        savepath = storage_folder + "_" + now

        return savepath

    def retrieve_training_and_testing_data(self, training: bool, training_set_percentage: int) -> (CustomDataset, CustomDataset):
        """Given the training percentage, retrieve the training and testing datasets.

        Args:
            training (bool): Flag to indicate whether the DataHandler will be used also for training (True) or for testing only (False)
            training_set_percentage (int): The percentage of the data to be used for training

        Returns:
            training_data (CustomDataset): The training dataset
            testing_data (CustomDataset): The testing dataset
        """

        # If training, create the path for the I/O-related storage
        if training:
            create_path(self.savepath + '/normalization')

        # ===================
        # RETRIEVE INPUT DATA
        # ===================

        # Initialize input vector
        X = []

        # Collect data from all the input paths
        for input_path in self.input_paths:
            with open(input_path, 'r') as openfile:
                current_input = json.load(openfile)
            X.extend(current_input)

        # Debug
        print("X size:", len(X), "x", len(X[0]))

        # Collect input statistics for denormalization
        X = np.asarray(X)
        Xmean, Xstd = X.mean(axis=0), X.std(axis=0)

        # Normalize input
        X_norm = normalize(X, axis=0)

        # Split into training inputs and test inputs
        splitting_index = round(training_set_percentage / 100 * X.shape[0])
        X_train = X_norm[:splitting_index]
        X_test = X_norm[splitting_index + 1:]

        # Debug
        print("X train size:", len(X_train), "x", len(X_train[0]))

        # If training, store input statistics (useful at inference time)
        if training:
            store_in_file(Xmean.tolist(), self.savepath + "/normalization/X_mean.txt")
            store_in_file(Xstd.tolist(), self.savepath + "/normalization/X_std.txt")

        # ====================
        # RETRIEVE OUTPUT DATA
        # ====================

        # Initialize output vector
        Y = []

        # Collect data from all output paths
        for output_path in self.output_paths:
            with open(output_path, 'r') as openfile:
                current_output = json.load(openfile)
            Y.extend(current_output)

        # Debug
        print("Y size:", len(Y), "x", len(Y[0]))

        # Collect output statistics for denormalization
        Y = np.asarray(Y)
        Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

        # Normalize output
        Y_norm = normalize(Y, axis=0)

        # Split into training outputs and test outputs
        Y_train = Y_norm[:splitting_index]
        Y_test = Y_norm[splitting_index + 1:]

        # Debug
        print("Y train size:", len(Y_train), "x", len(Y_train[0]))

        # If training, store output statistics (useful at inference time)
        if training:
            store_in_file(Ymean.tolist(), self.savepath + "/normalization/Y_mean.txt")
            store_in_file(Ystd.tolist(), self.savepath + "/normalization/Y_std.txt")

        # =====================
        # BUILD CUSTOM DATASETS
        # =====================

        training_data = CustomDataset(X_train, Y_train)
        testing_data = CustomDataset(X_test, Y_test)

        return training_data, testing_data

    def get_savepath(self) -> str:
        """Getter of the savepath.

        Returns:
            self.savepath (str): The path used for storage
        """

        return self.savepath

    def get_training_data(self):
        """Getter of the training dataset.

        Returns:
            self.training_data (CustomDataset): The training dataset
        """

        return self.training_data

    def get_testing_data(self):
        """Getter of the testing dataset.

        Returns:
            self.testing_data (CustomDataset): The testing dataset
        """

        return self.testing_data

