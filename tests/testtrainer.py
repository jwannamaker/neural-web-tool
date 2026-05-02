#!/usr/bin/env python
import pytest
import torch
from neuralwebtool.model import Network
from neuralwebtool.trainer import Trainer

class TestTrainer:
    def test_init(self):
        """Test that Trainer initializes correctly with a network and configuration."""
        model = Network([10,20,5])
        config = {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
        }
        trainer = Trainer(model, config)
        assert trainer.model is model 
        assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_init_invalid_loss(self):
        """Test trainer raised KeyError for invalid loss."""
        model = Network([10,5])
        config = {
            "loss": "invalid_loss",
            "optimizer": "adam",
            "lr": 0.0001,
        }
        with pytest.raises(KeyError):
            Trainer(model, config)
    
    def test_init_invalid_optimizer(self):
        """Test trainer raised KeyError for invalid optimizer."""
        model = Network([10,5])
        config = {
            "loss": "cross_entropy",
            "optimizer": "invalid_opt",
            "lr": 0.0001,
        }
        with pytest.raises(KeyError):
            Trainer(model, config)
    
    def test_train_step(self):
        """Test a single training step updates model parameters."""
        model = Network([10,20,5])
        config = {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001,
        }
        trainer = Trainer(model, config)

        #Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        #Create dummy data 
        image = torch.randn(4,10)
        labels = torch.randint(0,5,(4,))

        #Perform training step
        trainer.train_step(image, labels)

        #Check that parameters
        for initial, current in zip(initial_params, model.parameter()):
            assert not torch.equal(initial, current)