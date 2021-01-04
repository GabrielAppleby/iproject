from flask_restful import Resource, marshal_with, reqparse

from app.api.fields import data_instance_fields
from app.dao.wine_dao import WineDB

parser = reqparse.RequestParser()


class WineListAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self):
        return WineDB.query.all()


class WineAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self, uid):
        return WineDB.query.get_or_404(uid)
