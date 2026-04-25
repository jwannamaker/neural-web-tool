from app import app

@app.route('/')
@app.route('/index')
def index():
    return "Cheer up, baby, cheer up, baby"