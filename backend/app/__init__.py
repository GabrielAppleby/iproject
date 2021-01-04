import os

from flask import Flask
from flask_restful import Api
from flask_migrate import Migrate

from app.api.bacterium_api import BacteriumAPI, BacteriumListAPI
from app.api.flower_api import FlowerListAPI, FlowerAPI
from app.api.tf_model_api import TFModelAPI, TFShardAPI
from app.api.wine_api import WineListAPI, WineAPI
from app.dao.database import db


def create_app() -> Flask:
    app: Flask = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI=os.environ.get('DB_URI'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )

    db.init_app(app)
    migrate = Migrate(app, db)

    api: Api = Api(app, prefix='/api')

    api.add_resource(FlowerListAPI, '/flowers')
    api.add_resource(FlowerAPI, '/flowers/<int:uid>')

    api.add_resource(WineListAPI, '/wine')
    api.add_resource(WineAPI, '/wine/<int:uid>')

    api.add_resource(BacteriumListAPI, '/bacteria')
    api.add_resource(BacteriumAPI, '/bacteria/<int:uid>')

    api.add_resource(TFModelAPI, '/models/<string:dataset_name>')
    api.add_resource(TFShardAPI, '/models/<string:dataset_name>/<string:file_name>')

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET')

        return response

    return app


application = create_app()
