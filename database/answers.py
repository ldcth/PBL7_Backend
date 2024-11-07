from database.connect import database
from pymongo.errors import OperationFailure
import json
from bson import ObjectId
from datetime import datetime, timedelta


db = database['answers']


class Answers:
    def __init__(self, question="", answers=[], explanation="", correct_answer="", link=""):
        self.question = question
        self.answers = answers
        self.explanation = explanation
        self.correct_answer = correct_answer
        self.link = link
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

    def find_by_id(self, id):
        try:
            result = db.find_one({"_id": ObjectId(id)})
            if result:
                result['_id'] = str(result['_id'])
                self.answers = result['answers']
                self.question = result['question']
                self.explanation = result['explanation']
                self.correct_answer = result['correct_answer']
                self.link = result['link']
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    @staticmethod
    def get_total():
        try:
            res = db.count_documents({})
            return res
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None
        
    def find_by_question(self, question):
        try: 
            result = db.find_one({"question": {"$regex": question}})
            if result:
                result['_id'] = str(result['_id'])
                self.answers = result['answers']
                self.question = result['question']
                self.explanation = result['explanation']
                self.correct_answer = result['correct_answer']
                self.link = result['link']
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

    @staticmethod
    def get_data_months(current_year=datetime.utcnow().year):
        try:
            monthly_counts = [0] * 12
            for month in range(1, 13):
                # Tính toán ngày đầu tiên của tháng
                month_start = datetime(current_year, month, 1)
                # Tính toán ngày đầu tiên của tháng tiếp theo
                next_month_start = month_start + timedelta(days=31)
                # Lọc tài liệu trong khoảng thời gian từ ngày đầu tiên của tháng đến ngày đầu tiên của tháng tiếp theo
                count = db.count_documents(
                    {"createdAt": {"$gte": month_start, "$lt": next_month_start}})
                # Lưu số lượng tài liệu vào mảng
                monthly_counts[month - 1] = count
            return monthly_counts
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None
    
    @staticmethod
    def delete_by_month():
        try:
            db.delete_many({"createdAt": {"$gte": datetime(2024,4,1), "$lt": datetime(2024,4,30)}})
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None
