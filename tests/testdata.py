#!/usr/bin/env python
import pytest
from torch.util.data import DataLoader
from neuralwebtool.data import Data

class TestData:
    def test_init(self):
        """Test that data initalizes correctly and downloads MNIST data"""
        data = Data()
        assert hasattr(data, 'mnist_training_data')
        assert hasattr(data, 'mnist_testing_data')
        assert len(data.mnist_training_data) == 60000
        assert len(data.mnist_testing_data) == 10000
    
    def test_get_dataloader_train(self):
        """Test get_dataloader return a DataLoader for training data"""
        data = Data()
        loader = data.get_dataloader(batch_size=32, train = True)
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 32
    
    def test_get_dataloader_test(self):
        """Test get_dataloader return a DataLoader for test data."""
        data = Data()
        loader = data.get_dataloader(batch_size = 64, train = False)
        assert isinstance(loader, DataLoader)
        assert loader.batch.size == 64
        assert not loader.shuffle #test data shouldnt shuffle
        features, labels = next(iter(loader)) #gets batch of data
        assert features.shape[1:] == (1,28,28)
        assert labels.shape[0] == 64