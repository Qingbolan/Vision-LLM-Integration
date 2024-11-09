from flask import Blueprint, request, Response, stream_with_context
from flask import current_app
from flask import jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

import src.classfication


bp = Blueprint('chat',__name__,url_prefix='/chat')

# Load environment variables
load_dotenv('../../.env')

def extract_image_urls(attachment_data):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
    image_urls = []
    
    for item in attachment_data:
        file_url = item.get('file_url', '')
        if file_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_url = re.findall(pattern, file_url)
            if image_url:
                image_urls.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url[0]
                    }
                })
    
    return image_urls

def stream(input_text, attachments, history, official_Knwoledge, student_Knowledge):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
    
    print(attachments)
    
    image_urls = extract_image_urls(attachments)
    
    print(image_urls)
    
    messages = [
        {
            "role": "system",
            "content": f"""
            You are a Basic Deep Learning Model enhanced with AI Agent, use “Vision-LLM Integration” framework, designed by SILAN HU && Tan Kah Xuan.
            And this is a demo for the chatbot for NUS CS5242 Final Project.
            
            You can answer 
            
            1. 官方资料（优先考虑）：
            {official_Knwoledge}
            2. 学生问答记录：
            {student_Knowledge}
            """
        }
    ]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    user_content = image_urls + [{"type": "text", "text": input_text}]
    messages.append({"role": "user", "content": user_content})
    
    print(messages)

    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                full_response += text
                yield {"status": "processing", "chunk": text}
        
        yield {"status": "completed", "response": full_response}
    except Exception as e:
        print(f"Error in stream function: {str(e)}")
        yield {"status": "error", "message": str(e)}

@bp.route('/completion', methods=['POST'])
def completion():
    data = request.form
    if 'input_text' not in data:
        return jsonify({"error": "input_text is required"}), 400
    
    input_text = data['input_text']
    history = json.loads(data.get('history', '[]'))
    last_message_num = data.get('parent_message')
    course_num = data.get('course_number')
    attachments = json.loads(data.get('attachment', '[]'))
    chat_num = data.get('chat_number')
    
    ### debug -- info 
    print("=========================")
    print("input_text:", input_text)
    print("history:", history)
    print("last_message_num:", last_message_num)
    print("course_num:", course_num)
    print("attachments:", attachments)
    print("chat_num:", chat_num)
    print("=========================")

    # 简化的课程和教授信息
    professor_name = "Unknown"
    course_name = "Unknown"
    
    student_Knowledge = ""
    official_Knwoledge = ""
    
    def rag_official_knowledge():
        nonlocal official_Knwoledge
        nonlocal input_text
        print("正在官方知识树查找参考依据")
    
    def generate():
        nonlocal last_message_num
        nonlocal chat_num
        nonlocal student_Knowledge
        nonlocal official_Knwoledge
        
        full_response = ""
        
        try:
            yield f"data: {json.dumps({'status': 'Official_RAG', 'chunk': '正在官方知识树查找参考依据'})}\n\n"
            rag_official_knowledge()
            yield f"data: {json.dumps({'status': 'Official_RAG', 'chunk': f'找到{len(official_Knwoledge)}个相关的内容'})}\n\n"

            for result in stream(input_text, attachments, history, course_name, professor_name, official_Knwoledge, student_Knowledge):
                if result["status"] == "processing":
                    full_response += result["chunk"]
                    yield f"data: {json.dumps({'status': 'processing', 'chunk': result['chunk']})}\n\n"
                elif result["status"] == "completed":
                    full_response = result["response"]
                    break
                elif result["status"] == "error":
                    yield f"data: {json.dumps({'status': 'error', 'message': result['message']})}\n\n"
                    return
            
            official_Knwoledge = ""
            student_Knowledge = ""
            
            dl_analysis= ""
            
                    
            if chat_num is None or chat_num == 'temp':
                print(f"temp-compelete: {full_response}")

            yield f'''data: {json.dumps({'status': 'completed',
                                        'response': full_response,
                                        'index': { 
                                            'last_message_num': last_message_num,
                                            'chat_num': chat_num
                                        }})}\n\n'''
        except Exception as e:
            print(f"Error in generate function: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')


import os
import uuid
import random
import string
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_path(path):
    return secure_filename(path.strip().replace(' ', '_'))

def encrypt_file_id(file_id, key=17):
    encrypted_file_id = ''
    fixed_string = 'XoPlmABC'  # 仅包含英文字符
    for i, char in enumerate(str(file_id)):
        encrypted_char = chr((ord(char) - 48 + key + ord(fixed_string[i % len(fixed_string)])) % 36 + 48)
        if encrypted_char.isdigit():
            encrypted_file_id += encrypted_char
        else:
            encrypted_file_id += chr(ord(encrypted_char) + 7)
    
    encrypted_file_id = str(len(str(file_id))) + encrypted_file_id
    
    while len(encrypted_file_id) < 6:
        encrypted_file_id += random.choice(string.ascii_uppercase + string.digits)
    
    return encrypted_file_id

def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/')

def course_file_path(course, user, chat_num):
    course_directory = os.path.abspath(os.path.join(
        current_app.static_folder,
        'UserRegion',
        sanitize_path(course.course_num),
        sanitize_path(user.stu_num),
        sanitize_path(chat_num)
    ))
    course_directory = normalize_path(course_directory)
    print(f"Creating directory: {course_directory}")

    os.makedirs(course_directory, exist_ok=True)
    print(f"Directory created: {course_directory}")
    
    return course_directory

@bp.route('/api/upload/ChatFiles', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    chat_num = request.form.get('chat_num', 'temp')
    
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files allowed'}), 400
    
    for file in files:
        print('-------------------------')
        print('file_name:', file.filename)
        print('file_type:', file.content_type)
        file.seek(0, os.SEEK_END)
        print('file_size:', file.tell(), 'bytes')
        file.seek(0, 0)
        print('-------------------------')

    file_infos = []

    for file in files:
        if file.filename == '':
            continue

        if not allowed_file(file.filename):
            return jsonify({'error': f'Unsupported file type: {file.filename}'}), 400
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > 10 * 1024 * 1024:  # 10 MB limit
            return jsonify({'error': f'File too large: {file.filename}'}), 413

        ext = file.filename.rsplit('.', 1)[1].lower()
        raw_filename = sanitize_path(file.filename.rsplit('.', 1)[0])
        filename = 'raw_' + str(uuid.uuid4()) + '.' + ext
            
        file_path = os.path.join(
            '_uploads',
            filename
        )
        file_path = normalize_path(file_path)
        print(f"Attempting to save file: {file_path}")
    
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        try:
            file.save(file_path)
            print(f"File saved successfully: {file_path}")
        except IOError as e:
            print(f"Failed to save file: {file_path}. Error: {str(e)}")
            return jsonify({'error': f'Failed to save file: {file.filename}', 'details': str(e)}), 500
        
        file_url = f"{os.getenv('SERVER_BASE_URL')}{os.getenv('API_file_PREFIX')}_uploads/{filename}"
        print(file_url)
        
        file_infos.append({
            "file_name": raw_filename,
            "file_url": file_url,
            "file_type": file.content_type
        })
    
    # 简化日志记录
    print(f"Files uploaded successfully: {len(file_infos)} files")
    
    return jsonify({
        'success': True,
        'message': 'Files uploaded successfully',
        'data': {
            'files': file_infos,
            'file_num': len(file_infos),
            'chat_num': chat_num
        }
    })