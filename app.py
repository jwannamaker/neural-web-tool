"""Flask web application for neural network visualization and training."""
from flask import Flask, render_template, request, jsonify
import numpy as np
from Neuron import NeuralNetwork

app = Flask(__name__)
CURRENT_NETWORK = None


@app.route("/")
@app.route("/index")
def index():
    """Render the main index page."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Render the about page."""
    return "<p>This will be an about page<p>"


@app.route("/login", methods=["GET", "POST"])
def login():
    """Render the login page."""
    return "<p>This will be login page<p>"


@app.route("/user/<username>")
def profile(username):
    """Render the user profile page."""
    return f"<p>This will be a profile page for {username}<p>"


@app.route("/sandbox/")
def sandbox():
    """Render the sandbox page."""
    return "<p>This will be a sandbox<p>"


@app.route("/learn")
def learn():
    """Render the learning page."""
    return "<p>This will be the learning page<p>"


@app.route("/api/create_network", methods=["POST"])
def create_network():
    """Create a new neural network with specified layer sizes."""
    global CURRENT_NETWORK
    data = request.get_json()
    layer_sizes = data.get("layer_sizes")

    CURRENT_NETWORK = NeuralNetwork(layer_sizes)

    return jsonify({"message": "Network created successfully",
                    "layer_sizes": layer_sizes})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Make predictions using the current neural network."""
    if CURRENT_NETWORK is None:
        return jsonify({"error": "No network created yet"}), 400
    data = request.get_json()
    output = CURRENT_NETWORK.forward(np.array(data.get("input")))
    return jsonify({"output": output.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
