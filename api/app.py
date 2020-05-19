from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo

from . import settings
from .urls import app_routes


def init_app():

    app = Flask(__name__)
    app_routes(app)

    # Load Config File for DB
    app.config.from_pyfile(settings.CONFIGPATH)
    CORS(app)
    mongo = PyMongo(app)

    # Select the database
    db = mongo.db
    settings.shared_components["db"] = db
    return app


app = init_app()
