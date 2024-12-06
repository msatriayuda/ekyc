import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
GEMINI_MODEL = 'gemini-1.5-flash-001'