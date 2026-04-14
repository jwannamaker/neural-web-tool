"""Utilities for loading and processing MNIST dataset."""
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(base_url, data_sources, data_dir, request_opts=None):
    """Download datasets from remote URL.

    Args:
        base_url: Base URL for downloading data.
        data_sources: Dictionary mapping names to filenames.
        data_dir: Directory to save downloaded files.
        request_opts: Optional request parameters.
    """
    import requests

    if request_opts is None:
        request_opts = {}

    os.makedirs(data_dir, exist_ok=True)

    for filename in data_sources.values():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(
                base_url + filename, stream=True, **request_opts)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)


def images_to_ndarray(data_sources, data_dir, keys, mnist_dataset):
    """Convert image data files to numpy arrays.

    Args:
        data_sources: Dictionary mapping names to filenames.
        data_dir: Directory containing data files.
        keys: Keys to load from data_sources.
        mnist_dataset: Dictionary to populate with loaded images.

    Returns:
        Updated mnist_dataset with image arrays.
    """
    import gzip
    from typing import cast

    for key in keys:
        with gzip.open(
            os.path.join(data_dir, data_sources[key]), "rb") \
                as mnist_file:
            buffer = cast(bytes, mnist_file.read())
            mnist_dataset[key] = np.frombuffer(
                buffer, np.uint8, offset=16).reshape(-1, 28*28)

    return mnist_dataset


def labels_to_ndarray(data_sources, data_dir, keys, mnist_dataset):
    """Convert label data files to numpy arrays.

    Args:
        data_sources: Dictionary mapping names to filenames.
        data_dir: Directory containing data files.
        keys: Keys to load from data_sources.
        mnist_dataset: Dictionary to populate with loaded labels.

    Returns:
        Updated mnist_dataset with label arrays.
    """
    import gzip
    from typing import cast

    for key in keys:
        with gzip.open(
            os.path.join(data_dir, data_sources[key]), "rb") \
                as mnist_file:
            buffer = cast(bytes, mnist_file.read())
            mnist_dataset[key] = np.frombuffer(buffer, np.uint8, offset=8)

    return mnist_dataset


def plot_sample_images(x_train, image_index):
    """Display a sample image from the training set.

    Args:
        x_train: Training image data.
        image_index: Index of image to display.
    """
    mnist_image = x_train[image_index, :].reshape(28, 28)
    _, fig = plt.subplots()
    fig.imshow(mnist_image, cmap="gray")


def main():
    """Load and process MNIST dataset."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    data_sources = {
        "training_images": "train-images-idx3-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "training_labels": "train-labels-idx1-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    data_dir = "../data"
    load_data(base_url, data_sources, data_dir)

    mnist_dataset = {}
    images_to_ndarray(
        data_sources, "../data",
        ("training_images", "test_images"), mnist_dataset)
    labels_to_ndarray(
        data_sources, "../data",
        ("training_labels", "test_labels"), mnist_dataset)

    x_train, y_train = (mnist_dataset["training_images"],
                        mnist_dataset["training_labels"])
    x_test, y_test = (mnist_dataset["test_images"],
                      mnist_dataset["test_labels"])
    print(f"Shape of training images: {x_train.shape}, "
          f"{y_train.shape}")
    print(f"Shape of test images: {x_test.shape}, "
          f"{y_test.shape}")


if __name__ == "__main__":
    main()
