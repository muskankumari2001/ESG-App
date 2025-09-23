from flask import Flask
import os
from app.routes import bp as main_blueprint

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['DOWNLOAD_FOLDER'] = os.path.join(os.getcwd(), 'downloads')
    app.secret_key = 'secret'
    app.register_blueprint(main_blueprint)
    return app
