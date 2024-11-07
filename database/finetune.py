from database.connect import database
from pymongo.errors import OperationFailure
import json
from bson import ObjectId
from datetime import datetime, timedelta


db = database['finetune']


class Finetune:
    def __init__(self,  answers=0, result =""):
        self.answers = answers
        self.result = result
        self._id = None
        self.createdAt = datetime.utcnow()
        self.updatedAt = datetime.utcnow()

    def save_to_db(self):
        try:
            if self._id != None:
                self.updatedAt = datetime.utcnow()
                obj_dict = self.__dict__.copy()
                obj_dict.pop('_id', None)
                result = db.find_one_and_update({"_id": ObjectId(self._id)},
                                                {"$set": obj_dict})
            else:
                obj_dict = self.__dict__.copy()
                obj_dict.pop('_id', None)
                result = db.insert_one(obj_dict)
                self._id = str(result.inserted_id)
                return result.inserted_id
        except OperationFailure as e:
            print(f"Error save to database: {e}")
            return None
        
    @staticmethod
    def get_all():
        try:
            result = db.find({}).sort([('createdAt', -1)])
            if result:
                result = list(result)
                for item in result:
                    item["_id"] = str(item["_id"])
                    item["createdAt"] = str(item["createdAt"])
                    item["updatedAt"] = str(item["updatedAt"])
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error : {e}")
            return None

    def find_by_id(self, id):
        try:
            result = db.find_one({"_id": ObjectId(id)})
            if result:
                result['_id'] = str(result['_id'])
                self._id = str(result['_id'])
                self.answers = result['answers']
                self.result = result['result']
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    
    @staticmethod
    def delete_by_id(id):
        try:
            db.delete_one({"_id": ObjectId(id)})
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    