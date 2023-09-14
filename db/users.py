import pymongo
from devenv import MongoSecret
class Users:
    def __init__(self):
        self.myclient = pymongo.MongoClient(MongoSecret)
        self.mydb = self.myclient["myFirstDatabase"]
        self.mycol=self.mydb['VBA7users']
        print("User DB is active")
    
    def isUser(self,username):
        x = self.mycol.find_one({"username":username})
        if(x):
            return True
        else:
            return False


    def getUserHash(self,username):
        x = self.mycol.find_one({"username":username})
        if(x):
            return x["password"]
        else:
            return None

    def newUser(self,username,password):
        return self.mycol.insert_one({"username":username,"password":password})