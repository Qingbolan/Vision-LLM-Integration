from flask import Blueprint, request, Response, stream_with_context, jsonify, current_app
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import uuid
import logging
from werkzeug.utils import secure_filename
from src.llm.prompt import get_system_message

from src.utils.classfication import main as classification  # 确保路径正确

bp = Blueprint('chat', __name__, url_prefix='/chat')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv('../../.env')

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

# 检查文件是否被允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 安全化文件路径
def sanitize_path(path):
    return secure_filename(path.strip().replace(' ', '_'))

# 标准化路径
def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/')

# 全局临时存储字典
uploaded_images = []


# 提取图像URL的函数
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


chat_history = []  # 用于存储聊天历史

# 流式响应生成器
def stream(input_text, attachments, history, project_introduction):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))
    
    logging.info(f"Attachments: {attachments}")
    
    image_urls = extract_image_urls(attachments)
    
    logging.info(f"Image URLs: {image_urls}")
    
    messages = [
        {
            "role": "system",
            "content": f"""
You are an AI assistant with advanced image analysis capabilities.
You can assist with analyzing images and answering related queries.
{project_introduction}
"""
        }
    ]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 如果有上传的图片及分析结果，加入到消息中
    if input_text:
        messages.append({"role": "user", "content": input_text})
    
    logging.info(f"Messages: {messages}")

    try:
        response_stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        
        index=0
        full_response = ""
        for chunk in response_stream:
            # print(chunk)
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                full_response += text
                # print(text,end="")
                # print("\n\n=========================\n\n")
                index+=len(text)
                yield {"status": "processing", "chunk": text, "index": index}
        
        print("\n\n=========================\n\n")
        # print(full_response)
        yield {"status": "completed", "response": full_response}
    except Exception as e:
        logging.error(f"Error in stream function: {str(e)}")
        yield {"status": "error", "message": str(e)}

@bp.route('/completion', methods=['POST'])
def completion():
    data = request.form
    if 'input_text' not in data:
        return jsonify({"error": "input_text is required"}), 400
    
    input_text = data['input_text']
    history = json.loads(data.get('history', '[]'))
    chat_num = data.get('chat_number', 'temp')
    attachments = json.loads(data.get('attachment', '[]'))
    
    ### debug -- info 
    logging.info("=========================")
    logging.info(f"input_text: {input_text}")
    logging.info(f"history: {history}")
    logging.info(f"attachments: {attachments}")
    logging.info(f"chat_num: {chat_num}")
    logging.info("=========================")

    project_introduction = get_system_message()

    def generate():
        full_response = ""
        
        try:
            # 如果有上传的图片及分析结果，添加到上下文
            if uploaded_images:
                img_info = uploaded_images[-1]
                print("\n\n=========================\n\n")
                print(f"Uploaded image: {img_info['file_name']}\n\nAnalysis: {img_info['analysis']}")
                history.append({
                    "role": "system",
                    "content": f"Uploaded image: {img_info['file_name']}\n\nAnalysis: {img_info['analysis']}"
                })
                print("\n\n=========================\n\n")
                print(f"History: {history}")
                print("\n\n=========================\n\n")



            yield f"data: {json.dumps({'status': 'processing', 'chunk': ''})}\n\n"
            
            for result in stream(input_text, attachments, history, project_introduction):
                if result["status"] == "processing":
                    print(result['chunk'],end="")
                    if result['chunk']:
                        yield f"data: {json.dumps({'status': 'processing', 'chunk': result['chunk'], 'text_index': result['index'] })}\n\n"
                    # yield f"data: {json.dumps({'status': 'processing', 'chunk': result['chunk']})}\n\n"
                    
                if result["status"] == "completed":
                    full_response = result["response"]
                    break
                elif result["status"] == "error":
                    yield f"data: {json.dumps({'status': 'error', 'message': result['message']})}\n\n"
                    return
            
            yield f'''data: {json.dumps({'status': 'completed',
                                        'response': full_response,
                                        'index': { 
                                            'chat_num': chat_num
                                        }})}\n\n'''
        except Exception as e:
            logging.error(f"Error in generate function: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')


@bp.route('/api/upload/ChatFiles', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    chat_num = request.form.get('chat_num', 'temp')
    
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files allowed'}), 400
    
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
            'data',
            '_uploads',
            filename
        )
        file_path = normalize_path(file_path)
        logging.info(f"Attempting to save file: {file_path}")
    
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        try:
            file.save(file_path)
            logging.info(f"File saved successfully: {file_path}")
        except IOError as e:
            logging.error(f"Failed to save file: {file_path}. Error: {str(e)}")
            return jsonify({'error': f'Failed to save file: {file.filename}', 'details': str(e)}), 500
        
        file_url = f"{os.getenv('SERVER_BASE_URL')}{os.getenv('API_file_PREFIX')}_uploads/{filename}"
        logging.info(f"File URL: {file_url}")
        
        file_infos.append({
            "file_name": raw_filename,
            "file_url": file_url,
            "file_type": file.content_type
        })
    
    # 简化日志记录
    logging.info(f"Files uploaded successfully: {len(file_infos)} files")
    
    for file_info in file_infos:
        image_path = file_path  # 假设所有上传的文件路径都相同
        try:
            analysis_result = classification(
                image_path=image_path,
            )
        except Exception as e:
            logging.error(f"分析图片失败: {file_info['file_name']}, Error: {str(e)}")
            return jsonify({'error': f'Failed to analyze image: {file_info["file_name"]}', 'details': str(e)}), 500
        
        # 将分析结果与图片信息结合
        uploaded_images.append({
            'file_name': file_info['file_name'],
            'file_url': file_info['file_url'],
            'analysis': analysis_result
        })
        logging.info(f"Image analysis result: {analysis_result}")
    
    return jsonify({
        'success': True,
        'message': 'Files uploaded and analyzed successfully',
        'data': {
            'files': file_infos,
            'file_num': len(file_infos),
            'chat_num': chat_num
        }
    })
