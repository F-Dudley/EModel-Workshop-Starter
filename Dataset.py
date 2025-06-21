import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MaxAbsScaler


def load_data(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, tuple[MaxAbsScaler, MaxAbsScaler]]:
    """Load the dataset from CSV file and preprocess it.

    Args:
        file_path (str): _description_

    Returns:
        ndarray, ndarray: X and Y Values
        tuple: Scalers for X and Y values
    """

    assert os.path.exists(file_path), f"File '{file_path}' does not exist."

    df = pd.read_csv(file_path)
    df = df[df["Power"] < 1700]

    # Select the Relevant X and Y Features
    x_dataset = df[["Bitrate", "FileSize", "Quality", "Motion"]].copy()
    x_dataset["PixelRate"] = df["Height"] * df["Width"]
    y_dataset = df["Power"].to_frame()

    x_val = x_dataset.values
    y_val = y_dataset.values

    # Normalize the X and Y Values
    x_scaler = MaxAbsScaler()
    x_val = x_scaler.fit_transform(x_val)

    y_scaler = MaxAbsScaler()
    y_val = y_scaler.fit_transform(y_val)

    assert (
        x_val.shape[0] == y_val.shape[0]
    ), "X and Y values must have the same number of samples."

    return x_val, y_val, (x_scaler, y_scaler)
