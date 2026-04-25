#!/usr/bin/env python
from torch import float32
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize, Lambda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class Data:
    """
    A class for handling the MNIST dataset and applying necessary transformations to the images for training our model. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9) and their corresponding labels.

    We'll be using the `torchvision.transforms.v2` (2023) module as a drop-in replacement for the usual `torchvision.transforms` for performance gains and to better align with the PyTorch team's recommendations. (More info found here: https://docs.pytorch.org/vision/stable/transforms.html)

    First, we start with the PIL (Python Imaging Library) images from the MNIST dataset. (Notice, usually the ToTensor() method is first, but this is deprecated in favor of ToImage() and ToDtype(). More info here: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.v2.ToTensor)

    ToImage() converts the PIL image to a subclass of torch.Tensor without automatically scaling the pixel values to [0, 1].

    ToDtype(float32, scale=True) allows us to explicitly convert the tensor to float32 and scale the pixel values to [0, 1] by setting scale=True.

    Normalize(mean=[0.130627,], std=[0.308107,]) normalizes the pixel values using output[channel] = (input[channel] - mean[channel]) / std[channel]. We use these specific values based on the results from finding the mean and std of the MNIST dataset. We only need to pass in one value for both because the images are grayscaled. (More info here: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html#torchvision.transforms.v2.Normalize)

    Lambda(lambda x: ):
    """

    TRANSFORM = Compose(
        [
            ToImage(),
            ToDtype(float32, scale=True),
            Normalize(
                mean=[
                    0.130627,
                ],
                std=[
                    0.308107,
                ],
            ),
            # Explain why this needs to be squeezed in the first place, then analyze if needed
            # Lambda(lambda x: x.unsqueeze(0)),
        ]
    )
    
    def __init__(self):
        self.mnist_training_data = MNIST(root="./data", train=True, download=True, transform=Data.TRANSFORM)
        self.mnist_testing_data = MNIST(root="./data", train=False, download=True, transform=Data.TRANSFORM)

    def get_dataloader(self, batch_size: int, train: bool = True) -> DataLoader:
        """
        Loads the MNIST dataset and applies the defined transformations to the images. It returns a DataLoader that can be used for training or testing. ( More info on DataLoader here: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html )

        Args:
            batch_size (int): The number of samples per batch to load.
            train (bool): If True, loads the training set; otherwise, loads the test set.
        Returns:
            DataLoader: A DataLoader object that provides batches of transformed images and their corresponding labels.
        """
        return DataLoader(self.mnist_training_data if train else self.mnist_testing_data, batch_size=batch_size, shuffle=train)


# quick tests to see if the dataloader works
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch import randint

    data = Data()
    train_loader = data.get_dataloader(batch_size=64, train=True)
    test_loader = data.get_dataloader(batch_size=64, train=False)
    print(f"Training Loader: {train_loader}")
    print(f"Testing Loader: {test_loader}")

    train_features, train_labels = next(iter(train_loader))
    test_features, test_labels = next(iter(test_loader))
    print(f"Training feature batch shape: {train_features.size()}")
    print(f"Training labels batch shape: {train_labels.size()}")
    print(f"Testing feature batch shape: {test_features.size()}")
    print(f"Testing labels batch shape: {test_labels.size()}")

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        img, label = train_features[i], train_labels[i]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()