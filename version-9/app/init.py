# app/__init__.py
from flask import Flask

def create_app():
    """
    Application factory to create and configure the Flask app.
    """
    app = Flask(__name__)
    
    # Register routes
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
