from app import app
from flask import render_template, request, jsonify
from app.model import Network
from app.trainer import Trainer
from torchvision import datasets, transforms
import torch

current_network = None


# ----------------------
# MNIST LOADER
# ----------------------
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return dataset


# ----------------------
# ROUTES
# ----------------------
@app.route('/')
@app.route('/index')
def index():
    return render_template("network.html")


# ----------------------
# BUILD / PREDICT
# ----------------------
@app.route('/api/network', methods=['POST'])
def receive_network():
    global current_network

    data = request.json
    layer_sizes = data.get("neurons")

    if not layer_sizes:
        return jsonify({"status": "error", "message": "No layer data"}), 400

    try:
        activations = ["relu"] * (len(layer_sizes) - 1)

        current_network = Network(
            layer_sizes=layer_sizes,
            config={
                "loss": "cross_entropy",
                "optimizer": "adam",
                "lr": 0.001,
                "activations": activations,
            }
        )

        input_size = layer_sizes[0]

        dummy_input = torch.randn(1, input_size)
        output = current_network(dummy_input)

        # 🔥 convert to probabilities
        output = torch.softmax(output, dim=1)

        predicted = torch.argmax(output, dim=1).item()

        return jsonify({
            "status": "success",
            "output": output.detach().numpy().tolist(),
            "prediction": predicted,
            "weights": [
                layer.weight.detach().numpy().tolist()
                for layer in current_network.layers.values()
            ]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ----------------------
# TRAIN
# ----------------------
@app.route('/api/train', methods=['POST'])
def train_network():
    global current_network

    if current_network is None:
        return jsonify({"status": "error", "message": "No network created"}), 400

    try:
        dataset = load_mnist()

        input_size = current_network.layers["input"].in_features

        dataset = load_mnist()

        inputs = torch.stack([
           dataset[i][0][:input_size] for i in range(100)
        ])

        labels = torch.tensor([dataset[i][1] for i in range(100)])

        trainer = Trainer(current_network, {
            "loss": "cross_entropy",
            "optimizer": "adam",
            "lr": 0.001
        })

        for _ in range(100):
            trainer.train_step(inputs, labels)

        # 🔥 run prediction after training
        output = current_network(inputs[:1])

        # 🔥 convert to probabilities
        output = torch.softmax(output, dim=1)

        predicted = torch.argmax(output, dim=1).item()

        return jsonify({
            "status": "trained",
            "output": output.detach().numpy().tolist(),
            "prediction": predicted,
            "weights": [
                layer.weight.detach().numpy().tolist()
                for layer in current_network.layers.values()
            ]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500