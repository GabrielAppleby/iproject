from flask_restful import Resource

from app.dao.tf_model_dao import TFModelDB, TFShardDB


class TFModelAPI(Resource):

    def get(self, dataset_name):
        return TFModelDB.get_or_404(dataset_name)


class TFShardAPI(Resource):

    def get(self, dataset_name, file_name):
        return TFShardDB.get_or_404(dataset_name, file_name)
