from flask import Blueprint, request, Response, stream_with_context, jsonify, current_app
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import uuid
import logging
from werkzeug.utils import secure_filename

from src.utils.classfication import main as classification  # 确保路径正确

bp = Blueprint('chat', __name__, url_prefix='/chat')


logging.basicConfig(level=logging.DEBUG)

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
uploaded_images = {}

# 项目介绍信息（可根据需要修改或移除）
PROJECT_INTRODUCTION = """
研究目标：

构建一个可扩展的视觉分析框架，通过深度学习和大语言模型增强复杂视觉任务的处理能力
解决专业领域视觉分析中的自动化和智能化问题
提供一个统一的框架来处理不同领域的视觉分析需求

核心创新：

多模型融合架构：

监督学习模型：ResNet50、AlexNet、VGG16、ViT
无监督学习模型：Autoencoder、VAE、ViT Anomaly
自动化训练流程：

实现了端到端的自动化训练过程
智能模型选择和优化
支持不同专业领域的数据适配

LLM深度集成：

不是简单的API调用，而是深度融合
实现了自然语言交互界面
提供专业的视觉分析解释

技术架构：

训练阶段：

支持多领域数据输入（医疗、工业、科研等）
自动化的模型训练流程
完整的验证和评估系统

推理阶段：

支持多种视觉任务（分类、检测、分割）
智能化的结果分析
专业的报告生成

实验验证：

以裂缝检测为例进行了完整验证
在监督学习任务中取得了接近100%的高精度
在无监督学习任务中也达到了较高的性能

主要贡献：

提出了一个可扩展的视觉分析框架
创新性地结合了多种深度学习模型和LLM
实现了专业领域视觉分析的自动化和智能化
提供了完整的训练和推理解决方案

应用价值：

可以快速适应不同专业领域的需求
降低了专业视觉分析的使用门槛
提高了分析结果的可解释性
支持自然语言交互，提升用户体验

未来展望：

支持更多专业领域的扩展
进一步提升模型性能
增强系统的可解释性
优化自动化训练流程

这个项目的核心价值在于：

提供了一个统一的框架来解决专业领域的视觉分析问题
通过多模型融合和LLM集成提升了系统性能和易用性
实现了从数据到结果的全流程自动化
具有很强的实用价值和扩展性
"""

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
{project_introduction}
You can assist with analyzing images and answering related queries.
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
            model="gpt-4",
            messages=messages,
            stream=True,
        )
        
        full_response = ""
        for chunk in response_stream:
            if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0]:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    text = delta['content']
                    full_response += text
                    yield {"status": "processing", "chunk": text}
        
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

    project_introduction = PROJECT_INTRODUCTION  # 可以根据需要修改或移除

    def generate():
        full_response = ""
        
        try:
            # 如果有上传的图片及分析结果，添加到上下文
            if chat_num in uploaded_images:
                for img_info in uploaded_images[chat_num]:
                    analysis_text = ""
                    if 'predicted_label' in img_info['analysis']:
                        analysis_text = f"图片 '{img_info['file_name']}' 的分类结果为: {img_info['analysis']['predicted_label']}，置信度: {img_info['analysis']['probability']*100:.2f}%."
                    elif 'anomaly_score' in img_info['analysis']:
                        analysis_text = f"图片 '{img_info['file_name']}' 的异常得分为: {img_info['analysis']['anomaly_score']:.4f}。"
                    elif 'reconstruction_error' in img_info['analysis']:
                        analysis_text = f"图片 '{img_info['file_name']}' 的重构误差为: {img_info['analysis']['reconstruction_error']:.4f}。"
                    messages = {"role": "user", "content": analysis_text}
                    history.append(messages)

            yield f"data: {json.dumps({'status': 'processing', 'chunk': '正在处理您的请求...'})}\n\n"
            
            for result in stream(input_text, attachments, history, project_introduction):
                if result["status"] == "processing":
                    yield f"data: {json.dumps({'status': 'processing', 'chunk': result['chunk']})}\n\n"
                elif result["status"] == "completed":
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
    
    # 如果有上传的文件，调用分析模块并存储结果
    if chat_num not in uploaded_images:
        uploaded_images[chat_num] = []
    
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
        uploaded_images[chat_num].append({
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
