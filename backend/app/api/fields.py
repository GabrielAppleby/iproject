from typing import Dict

import numpy as np
from flask_restful import fields


class BytesField(fields.Raw):
    def format(self, value):
        return np.frombuffer(value, dtype=np.float32).tolist()


data_instance_fields: Dict = {
    "uid": fields.Integer,
    "features": BytesField,
    "target": fields.String
}
