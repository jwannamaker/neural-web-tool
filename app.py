"""Flask web application for neural network visualization and training."""
from flask import Flask, render_template, request, jsonify
<<<<<<< HEAD

app = Flask(__name__)
current_network = None
=======
import numpy as np
from neural_network import NeuralNetwork

app = Flask(__name__)
CURRENT_NETWORK = None
>>>>>>> main


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
