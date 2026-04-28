from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/index")
def index_alias():
    return render_template("index.html")


@app.route("/about")
def about():
    return "about"


@app.route("/network")
def network():
    return render_template("network.html")


with app.test_request_context():
    try:
        print("--- INDEX ---")
        print(render_template("index.html"))
        print("--- NETWORK ---")
        print(render_template("network.html"))
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
