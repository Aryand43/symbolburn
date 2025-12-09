from dotenv import load_dotenv
import os

load_dotenv()

LANGDB_API_KEY = os.getenv("LANGDB_API_KEY")
LANGDB_PROJECT_ID = os.getenv("LANGDB_PROJECT_ID")
