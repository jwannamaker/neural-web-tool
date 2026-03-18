from flask import Flask, url_for

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return "<p>This is the base URL<p>"


@app.route("/about")
def about():
    return "<p>This will be an about page<p>"


@app.route("/login", methods=["GET", "POST"])
def login():
    return (
        "<p>A login page for anyone to use their learning material or make new ANNs<p>"
    )


@app.get("/login")
def login_get():
    return "<p>Show a login page here<p>"


@app.post("/login")
def login_post():
    return "<p>Do the logging in<p>"


@app.route("/user/<username>")
def profile(username):
    return f"<p>This is a profile page for {username}'s dashboard or whatever<p>"


@app.route("/sandbox/")
def playground():
    return "<p>This will be where the actual ANN playgrounds go<p>"


def test_urls():
    with app.test_request_context():
        print(url_for("index"))
        print(url_for("login"))
        print(url_for("playground"))


if __name__ == "__main__":
    test_urls()
