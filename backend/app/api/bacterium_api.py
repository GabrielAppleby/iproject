from flask_restful import Resource, marshal_with, reqparse

from app.api.fields import data_instance_fields
from app.dao.bacterium_dao import BacteriumDB

parser = reqparse.RequestParser()


class BacteriumListAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self):
        return BacteriumDB.query.all()


class BacteriumAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self, uid):
        return BacteriumDB.query.get_or_404(uid)
