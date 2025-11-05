from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = '123456789'

    with app.app_context():
        from . import routes  # Import routes

    return app
