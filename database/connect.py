from pymongo import MongoClient

uri = "mongodb+srv://NgVSang:Sang100302@demoproject.mbabi.mongodb.net/?retryWrites=true&w=majority&appName=demoProject"

client = MongoClient(uri)

database = client.get_database('PBL_7')
