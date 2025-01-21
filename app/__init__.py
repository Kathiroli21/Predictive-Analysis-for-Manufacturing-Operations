from app.routes.api_routes import api_bp
from flask import Flask
def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.register_blueprint(api_bp)
    return app
