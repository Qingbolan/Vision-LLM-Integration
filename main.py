from flask import Flask
from app import create_app
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv('.env')

if __name__ == "__main__":    
    app = Flask(__name__, static_folder='data/', static_url_path=os.getenv('API_file_PREFIX'))
    app = create_app(app)
    CORS(app, supports_credentials=True)
    app.run(debug=True, host='0.0.0.0', port=5100)  # Run the app