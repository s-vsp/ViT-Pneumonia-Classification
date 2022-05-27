"""Utilities for exploration of pneumonia dataset, both with training and evaluating a model.

Content:

- load_pneumonia
- visualize_data
- image_wrangling
"""

import os
import pathlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_pneumonia(path, *args, **kwargs) -> dict:
    """Loads full pneumonia dataset (train / val / test).
    Images are of shape X x X, and the given labels are: 1 - PNEUMONIA, 0 - NORMAL.

    Args:
        - path: path to main data folder with data 
    
    Returns:
        - data_dict: {
            "train": [(x1_train, y1_train), ..., (xn_train, yn_train)]: images and labels (training data)
            "val": [(x1_val, y1_val), ..., (xn_val, yn_val)]: images and labels (validation data)
            "test": [(x1_test, y1_test), ..., (xn_test, yn_test)]: images and labels (testing data)
            }: data dictionary, where keys define affiliation (train / val / test) and values are lists of tuples (images, labels) 
    """
    data_dir = pathlib.Path(path)

    # TRAIN
    train_dir = data_dir / "train"

    normal_train_dir = train_dir / "NORMAL"
    pneumonia_train_dir = train_dir / "PNEUMONIA"

    normal_train_images = normal_train_dir.glob("*.jpeg")
    pneumonia_train_images = pneumonia_train_dir.glob("*.jpeg")
    
    train_data = list()

    for image in normal_train_images:
        train_data.append((plt.imread(image), 0))
    for image in pneumonia_train_images:
        train_data.append((plt.imread(image),1))
    
    # VALIDATION
    val_dir = data_dir / "val"

    normal_val_dir = val_dir / "NORMAL"
    pneumonia_val_dir = val_dir / "PNEUMONIA"

    normal_val_images = normal_val_dir.glob("*.jpeg")
    pneumonia_val_images = pneumonia_val_dir.glob("*.jpeg")

    val_data = list()

    for image in normal_val_images:
        val_data.append((plt.imread(image),0))
    for image in pneumonia_val_images:
        val_data.append((plt.imread(image),1))
    
    # TEST
    test_dir = data_dir / "test"

    normal_test_dir = test_dir / "NORMAL"
    pneumonia_test_dir = test_dir / "PNEUMONIA"

    normal_test_images = normal_test_dir.glob("*.jpeg")
    pneumonia_test_images = pneumonia_test_dir.glob("*.jpeg")

    test_data = list()

    for image in normal_test_images:
        test_data.append((plt.imread(image),0))
    for image in pneumonia_test_images:
        test_data.append((plt.imread(image),1))
    
    # FINAL DATA DICT
    data_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    return data_dict

def visualize_data(data, batch_size=4, *args, **kwargs):
    """Visualizes a random batch of images from a specified dataset.

    Args:
        - data: dataset of images being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
        - batch_size: number of images to be visualized (default is 4)
    
    Returns:
        - None
    """
    idxs = np.random.randint(0, len(data), batch_size)

    labels = ["NORMAL", "PNEUMONIA"]

    plt.figure(figsize=(14,7))
    for i, idx in enumerate(idxs):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(data[idx][0], cmap="gray")
        plt.title(labels[data[idx][1]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def image_wrangling(data: list, output_shape: tuple, *args, **kwargs) -> list:
    """Rescales and expand dims (number of image channels). Due to the most efficent way to process images into the
    model, following suggestions of different authors of different papers, appropriate operations on images are 
    needed.

    Args:
        - data: dataset of images being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
        - output_shape: tuple representing -> heigth (x-axis), width (y-axis)
    
    Returns:
        - rescaled_data: dataset of images after operations being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
    """
    rescaled_data = []
    for image in data:
        new_image = image[0]
        label = image[1]
        new_image = cv2.resize(new_image, output_shape)
        new_image = np.dstack([new_image, new_image, new_image])
        rescaled_data.append((new_image, label))

    return rescaled_data

def image_normalize(data: list, *args, **kwargs) -> list:
    """Normalizes images to the (0.0 - 1.0) scale.

    Args:
        - data: dataset of images being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
    
    Returns:
        - normalized_data: dataset of normalized images being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
    """
    normalized_data = []
    for image in data:
        new_image = image[0]
        label = image[1]
        new_image = (new_image / 255.).astype(np.float32)
        normalized_data.append((new_image, label))
    
    return normalized_data

def extract_by_label(data, label, *args, **kwargs) -> list:
    """Extracts the images from a dataset associated only with the given label.

    Args:
        - data: dataset of images being a list structure: [(image_1, label_1), ..., (image_n, label_n)]
        - label: image class belonging
    
    Returns:
        - extracted_data: list of images: [numpy.array(image_1), ..., numpy.array(image_n)]
    """
    extracted_images = []
    for image in data:
        if image[1] == label:
            extracted_images.append(image[0])
        else:
            continue

    return extracted_images


if __name__ == "__main__":
    path = ".//data//chest_xray//chest_xray"
    data = load_pneumonia(path)
    train = data["train"]
    val = data["val"]
    test = data["test"]
    print(f"train {len(train)}")
    print(f"val {len(val)}")
    print(f"test {len(test)}")