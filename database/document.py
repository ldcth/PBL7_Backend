from database.connect import database
from pymongo.errors import OperationFailure
import json
from bson import ObjectId
from datetime import datetime


db = database['documents']


class Document:
    def __init__(self, text=None, embedding=[], book_id=None):
        self.text_preprocessed_vietnamese = text
        self.embedding = embedding
        self.book_id = book_id
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

    def find_by_book_id(self, book_id):
        try:
            result = db.find({"book_id": book_id})
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
                self.text_preprocessed_vietnamese = result['text_preprocessed_vietnamese']
                self.embedding = result['embedding']
                self.book_id = result['book_id']
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
                self.text_preprocessed_vietnamese = result['text_preprocessed_vietnamese']
                self.embedding = result['embedding']
                self.book_id = result['book_id']
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None
