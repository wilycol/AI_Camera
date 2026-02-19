import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB Connection Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ai_camera_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
faces_collection = db["faces"]  # Stores label and embedding
logs_collection = db["logs"]    # Stores recognition events
