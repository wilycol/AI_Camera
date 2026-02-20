import os
import json
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
LOCAL_DB_FILE = "local_db.json"

class LocalCollection:
    def __init__(self, filename, collection_name):
        self.filename = filename
        self.collection_name = collection_name
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                json.dump({}, f)

    def _load(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return data.get(self.collection_name, [])
        except:
            return []

    def _save(self, data_list):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
            data[self.collection_name] = data_list
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        except:
            pass

    def find(self, query=None, projection=None):
        data = self._load()
        if not query:
            return data
        # Very basic filtering
        results = []
        for item in data:
            match = True
            for k, v in query.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                results.append(item)
        return results

    def update_one(self, filter, update, upsert=False):
        data = self._load()
        target = None
        for i, item in enumerate(data):
            match = True
            for k, v in filter.items():
                if item.get(k) != v:
                    match = False
                    break
            if match:
                target = i
                break

        updates = update.get("$set", {})
        on_insert = update.get("$setOnInsert", {})

        if target is not None:
            data[target].update(updates)
        elif upsert:
            new_item = filter.copy()
            new_item.update(on_insert)
            new_item.update(updates)
            data.append(new_item)
        
        self._save(data)

    def insert_one(self, document):
        data = self._load()
        data.append(document)
        self._save(data)

# Toggle between MongoDB and Local JSON
if MONGO_URI and not MONGO_URI.startswith("mongodb://localhost"):
    print("--- Using MongoDB Atlas ---")
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    db = client["ai_camera_db"]
    faces_collection = db["faces"]
    logs_collection = db["logs"]
else:
    print(f"--- Using Local JSON Storage ({LOCAL_DB_FILE}) ---")
    faces_collection = LocalCollection(LOCAL_DB_FILE, "faces")
    logs_collection = LocalCollection(LOCAL_DB_FILE, "logs")
