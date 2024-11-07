from database.connect import database
from pymongo.errors import OperationFailure
import json
from bson import ObjectId
from datetime import datetime, timedelta
from flask import jsonify


db = database['links']


class Links:
    def __init__(self, name="", questions=[]):
        self.name = name
        self.questions = questions
        self._id = None
        self.new = True
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
                # self._id = str(result.inserted_id)
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
    def get_all_links(page=None, per_page=None):
        try:
            if page and per_page:
                skip = (page - 1) * per_page
                result = db.find({}).sort([('createdAt', -1)]).skip(skip).limit(per_page)
                if result:
                    result = list(result)
                    for item in result:
                        item["_id"] = str(item["_id"])
                        # questions = []
                        # for data in item['questions']:
                        #     questions.append(str(data))
                        item["questions"] = len(item["questions"])
                        item["createdAt"] = str(item["createdAt"])
                        item["updatedAt"] = str(item["updatedAt"])
                    return result
                else:
                    return None
            else:
                result = db.find({}).sort([('createdAt', -1)])
                if result:
                    result = list(result)
                    for item in result:
                        item["_id"] = str(item["_id"])
                        # questions = []
                        # for data in item['questions']:
                        #     questions.append(str(data))
                        item["questions"] = len(item["questions"])
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
                # self = jsonify(result)
                self._id = str(result['_id'])
                self.name = result['name']
                self.questions = result['questions']
                self.new = result['new']
                self.createdAt = result['createdAt']
                self.updatedAt = result['updatedAt']
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error finding: {e}")
            return None

    @staticmethod
    def delete_by_id(id):
        try:
            db.delete_one({"_id": ObjectId(id)})
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    @staticmethod
    def check_name_exis(name):
        try:
            result = db.find_one({"name": name})
            if result:
                return True
            else:
                return False
        except OperationFailure as e:
            return None

    @staticmethod
    def get_total():
        try:
            res = db.count_documents({})
            return res
        except OperationFailure as e:
            print(f"Error finding settings: {e}")
            return None

    @staticmethod
    def get_pending():
        try:
            pipeline = [
                {"$match": {"new": True}},  # Lọc tài liệu có new = True
                # Trích xuất độ dài của mảng questions
                {"$project": {"questions_length": {"$size": "$questions"}}},
                {"$group": {"_id": None, "total_questions_length": {
                    "$sum": "$questions_length"}}}  # Tính tổng độ dài
            ]
            result = list(db.aggregate(pipeline))
            total_questions_length = result[0]['total_questions_length'] if result else 0
            return total_questions_length
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
        
    @staticmethod
    def get_lated_data():
        try:
            result = db.find({}).sort([('createdAt', -1)]).skip(0).limit(50)
            if result:
                result = list(result)
                for item in result:
                    item["_id"] = str(item["_id"])
                    item["questions"] = len(item["questions"])
                    item["createdAt"] = str(item["createdAt"])
                    item["updatedAt"] = str(item["updatedAt"])
                return result
            else:
                return None
        except OperationFailure as e:
            print(f"Error : {e}")
            return None
