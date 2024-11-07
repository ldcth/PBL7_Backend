from bson import ObjectId
from flask import jsonify


def custom_json_respone(obj: object):
    return {
        **obj,
        '_id': str(obj['_id'])
    }
