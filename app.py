from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return "<p>This will be an about page<p>"


@app.route("/login", methods=["GET", "POST"])
def login():
    return "<p>This will be login page<p>"


@app.route("/user/<username>")
def profile(username):
    return f"<p>This will be a profile page for {username}<p>"


@app.route("/sandbox/")
def sandbox():
    return "<p>This will be a sandbox<p>"


@app.route("/learn")
def learn():
    return "<p>This will be the learning page<p>"


# def test_urls():
#     with app.test_request_context():
#         print(url_for("index"))
#         print(url_for("login"))
#         print(url_for("playground"))
#

if __name__ == "__main__":
    # test_urls()
    app.run(host="0.0.0.0", port=8000)
