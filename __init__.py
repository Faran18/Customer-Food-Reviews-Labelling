from flask import Flask
import os

def create_app():
    app = Flask(__name__, template_folder='templates')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Import and register routes
    from .routes import init_routes
    init_routes(app)

    return app