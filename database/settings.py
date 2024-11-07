from database.connect import database
from pymongo.errors import OperationFailure
import json
from bson import ObjectId
from datetime import datetime


db = database['settings']


class Settings:
    def __init__(self, crawl=None, type=None, date=None, time=None):
        self.crawl = crawl
        self.type = type
        self.date = date
        self._id = None
        self.time = time
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

    def find_by_name(self, name):
        try:
            result = db.find({"name": name})
            if result:
                result = list(result)
                for item in result:
                    item['_id'] = str(item['_id'])
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    def find_by_id(self, id):
        try:
            result = db.find_one({"_id": ObjectId(id)})
            if result:
                result['_id'] = str(result['_id'])
                self.crawl = result['crawl']
                self.date = result['date']
                self.type = result['type']
                self.time = result['time']
                self._id = str(result['_id'])
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    def find_one(self):
        try:
            result = db.find_one({})
            if result:
                result['_id'] = str(result['_id'])
                self._id = str(result['_id'])
                self.crawl = result['crawl']
                self.date = result['date']
                self.type = result['type']
                self.time = result['time']
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None
