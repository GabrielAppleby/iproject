from flask_restful import Resource, marshal_with, reqparse

from app.api.fields import data_instance_fields
from app.dao.flower_dao import FlowerDB

parser = reqparse.RequestParser()


class FlowerListAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self):
        return FlowerDB.query.all()


class FlowerAPI(Resource):
    @marshal_with(data_instance_fields)
    def get(self, uid):
        return FlowerDB.query.get_or_404(uid)
