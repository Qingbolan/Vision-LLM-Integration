from flask import Flask
from app import create_app
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv('.env')

if __name__ == "__main__":    
    app = Flask(__name__, static_folder='data', static_url_path=os.getenv('API_file_PREFIX'))
    # 打印Flask应用的静态文件配置
    app = create_app(app)
    print("=== Flask App Static Files Info ===")
    print(f"Flask Static Folder: {app.static_folder}")
    print(f"Flask Static URL Path: {app.static_url_path}")
    print("================================\n")
    CORS(app, supports_credentials=True)
    app.run(debug=True, host='0.0.0.0', port=5100)  # Run the app