"""Flask web application for neural network visualization and training."""
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import io
from PIL import Image
import numpy as np
import torch
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

from neuralwebtool.model import Network as NeuralNetwork
from neuralwebtool.trainer import Trainer
from neuralwebtool.data import Data

app: Flask = Flask(__name__)
current_network: Optional[NeuralNetwork] = None
training_logs: List[Dict[str, str]] = []

def add_log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    training_logs.append({
        "timestamp": timestamp,
        "level": level,
        "message": msg
    })
    # Keep only last 500 logs to prevent memory bloat
    if len(training_logs) > 500:
        training_logs.pop(0)


@app.route("/")
@app.route("/index")
def index() -> str:
    """Render the main index page."""
    return render_template("index.html")


@app.route("/about")
def about() -> str:
    """Render the about page."""
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    """Render the login page."""
    return "<p>This will be login page<p>"


@app.route("/user/<username>")
def profile(username: str) -> str:
    """Render the user profile page."""
    return f"<p>This will be a profile page for {username}<p>"


@app.route("/sandbox/")
def sandbox() -> str:
    """Render the sandbox page."""
    return render_template("sandbox.html")


@app.route("/help")
def help_page() -> str:
    """Render the Help Topics page."""
    return render_template("help.html")


@app.route("/logs")
def logs_page() -> str:
    """Render the Training Logs page."""
    return render_template("logs.html")


@app.route("/api/logs_data")
def logs_data() -> Union[Response, Tuple[Response, int]]:
    """Return the current training logs."""
    return jsonify({"logs": training_logs})


@app.route("/learn")
def learn() -> str:
    return "<p>This will be the learning page<p>"


@app.route("/network")
def network() -> str:
    return render_template("network.html")


@app.route("/api/create_network", methods=["POST"])
def create_network() -> Union[Response, Tuple[Response, int]]:
    data: Optional[Dict[str, Any]] = request.get_json()
    
    if data is None:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400
        
    layer_sizes: List[int] = data.get("layer_sizes", [])

    # create and store network
    global current_network
    current_network = NeuralNetwork(layer_sizes)

    return jsonify({"message": "Network created successfully", "layer_sizes": layer_sizes})


@app.route("/api/predict", methods=["POST"])
def predict() -> Union[Response, Tuple[Response, int]]:
    global current_network
    if current_network is None:
        return jsonify({"error": "No network created yet"}), 400
        
    data: Optional[Dict[str, Any]] = request.get_json()
    if data is None or "input" not in data:
        return jsonify({"error": "Invalid or missing input data"}), 400
    
    # Convert input to tensor and reshape
    input_list: List[float] = data.get("input", [])
    input_data: torch.Tensor = torch.tensor(input_list, dtype=torch.float32)
    
    # Ensure it has batch dimension
    if len(input_data.shape) == 1:
        input_data = input_data.unsqueeze(0)
        
    with torch.no_grad():
        output: torch.Tensor = current_network(input_data)
        
    return jsonify({"output": output.tolist()})


@app.route("/api/train", methods=["POST"])
def train() -> Union[Response, Tuple[Response, int]]:
    global current_network
    req_data: Optional[Dict[str, Any]] = request.get_json()
    
    if req_data is None:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400
    
    layer_sizes: List[int] = req_data.get("layer_sizes", [784, 128, 64, 10])
    activations: List[str] = req_data.get("activations", ["relu", "relu", "linear"])
    loss: str = req_data.get("loss", "cross_entropy")
    optimizer: str = req_data.get("optimizer", "adam")
    lr: float = req_data.get("lr", 0.001)
    epochs: int = req_data.get("epochs", 1)
    batch_size: int = req_data.get("batch_size", 64)
    
    if len(activations) != len(layer_sizes) - 1:
        return jsonify({"error": "Number of activation functions must be one less than number of layers."}), 400

    config: Dict[str, Any] = {
        "loss": loss,
        "optimizer": optimizer,
        "lr": lr,
        "activations": activations,
    }
    
    current_network = NeuralNetwork(layer_sizes=layer_sizes, config=config)
    data_handler: Data = Data()
    train_loader: Any = data_handler.get_dataloader(batch_size=batch_size, train=True)
    trainer: Trainer = Trainer(current_network, config)
    
    global training_logs
    training_logs.clear()
    add_log(f"Initializing Neural Network with architecture: {layer_sizes}")
    add_log(f"Config: Loss={loss}, Optimizer={optimizer}, LR={lr}")
    add_log(f"Loading MNIST dataset (Batch Size: {batch_size})...")
    
    # Simple synchronous training loop for the sandbox
    for epoch in range(epochs):
        add_log(f"--- Starting Epoch {epoch+1}/{epochs} ---", "INFO")
        batch_idx = 0
        for images, labels in train_loader:
            images = images.reshape(images.size(0), -1)
            trainer.train_step(images, labels)
            
            # Log every 100 batches to avoid flooding
            if batch_idx % 100 == 0:
                add_log(f"Epoch {epoch+1} | Processed batch {batch_idx}...", "DEBUG")
            batch_idx += 1
            
        add_log(f"Epoch {epoch+1} Complete.", "SUCCESS")
        
    add_log(f"Training successfully completed for {epochs} epochs.", "SUCCESS")
            
    return jsonify({"message": f"Training complete for {epochs} epochs!"})


@app.route("/api/evaluate", methods=["POST"])
def evaluate() -> Union[Response, Tuple[Response, int]]:
    global current_network
    if current_network is None:
        return jsonify({"error": "No network created/trained yet."}), 400
        
    req_data: Optional[Dict[str, Any]] = request.get_json()
    batch_size: int = 64
    
    if req_data is not None:
        batch_size = req_data.get("batch_size", 64)
    
    data_handler: Data = Data()
    test_loader: Any = data_handler.get_dataloader(batch_size=batch_size, train=False)
    
    current_network.eval()
    correct: int = 0
    total: int = 0
    
    # Track per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(images.size(0), -1)
            outputs: torch.Tensor = current_network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
    accuracy: float = 100.0 * correct / total
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
            
    current_network.train() # Set back to train mode just in case
    
    return jsonify({
        "accuracy": accuracy,
        "class_accuracies": class_accuracies
    })


@app.route("/api/predict_image", methods=["POST"])
def predict_image() -> Union[Response, Tuple[Response, int]]:
    global current_network
    if current_network is None:
        return jsonify({"error": "No network created yet"}), 400
        
    data: Optional[Dict[str, Any]] = request.get_json()
    if data is None or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400
    
    try:
        # Decode base64 image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        # MNIST images are black background (0) and white text (255)
        # We need to make sure the input matches this.
        # Usually drawings on canvas are black on white, so we might need to invert.
        # The user's drawing logic will determine this.
        img_array = np.array(image, dtype=np.float32)
        
        # Simple normalization to [0, 1] then match MNIST stats if possible
        # Or just use the Data.TRANSFORM logic
        # But here we'll do it manually for simplicity
        img_array = img_array / 255.0
        
        # MNIST is white on black. If canvas is black on white, invert:
        # We'll assume the frontend sends white on black or we invert here.
        # Let's assume frontend sends white stroke on black background.
        
        # Apply normalization matching Data.py
        # mean=[0.130627,], std=[0.308107,]
        img_array = (img_array - 0.130627) / 0.308107
        
        # Flatten and convert to tensor
        input_tensor = torch.tensor(img_array).reshape(1, 784)
        
        current_network.eval()
        with torch.no_grad():
            output = current_network(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
        probs_list = probabilities.tolist()
        top_indices = np.argsort(probs_list)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "digit": int(idx),
                "confidence": float(probs_list[idx])
            })
            
        return jsonify({"predictions": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # check for a port argument
    port: int = 8080
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    
    app.run(host="0.0.0.0", port=port, debug=True)