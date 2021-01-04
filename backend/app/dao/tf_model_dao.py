import json
import os

from flask import send_from_directory
from werkzeug.exceptions import abort

DAO_FOLDER: str = os.path.dirname(os.path.realpath(__file__))


class TFModelDB:

    @staticmethod
    def get_or_404(dataset_name):
        model_path = os.path.join(DAO_FOLDER, 'tf_models', dataset_name, 'model.json')
        print(model_path)
        with open(model_path, 'r') as model:
            if model is None:
                abort(404)
            return json.load(model)


class TFShardDB:

    @staticmethod
    def get_or_404(dataset_name, file_name):
        shard_path = os.path.join(DAO_FOLDER, 'tf_models', dataset_name)
        return send_from_directory(shard_path, file_name)
