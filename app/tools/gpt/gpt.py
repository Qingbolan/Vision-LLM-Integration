from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def callGPT(messages) -> str:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE') or None)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
    )
    print()
    print(chat_completion.choices[0].message.content)
    print()
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    print(asyncio.run(callGPT(
        [
            {
                "role": "user",
                "content": "请用python写一个贪吃蛇",
            }
        ]
    )))
    
    
# from flask import request, Response, stream_with_context
# from flask_jwt_extended import jwt_required, get_jwt_identity
# from app.models import User, Course, Chat, ChatMessage
# from app import db

# @bp.route('/completion', methods=['POST'])
# @jwt_required()
# def completion_api():
#     if request.method == "POST":
#         data = request.form  # 获取表单数据
#         if 'input_text' not in data:
#             return Response("input_text key is missing", status=400)
#         input_text = data['input_text']
#         history = data.get('history', '')  # 如果没有提供历史记录,默认为空字符串
#         chat_num = data.get('chat_number')
#         course_num = data.get('course_number')
#         print(data)

#         attachments = data.get('attachment')

#         print(input_text)
#         print(type(history))
#         print(attachments)
        
#         current_user = get_jwt_identity()
#         user = User.query.filter_by(stu_num=current_user).first()
        
#         professor_name = user.username
#         course_name = "测试课程"
#         course = Course.query.filter_by(course_num=course_num).first()
        
#         professor_name = User.query.filter_by(user_id=course.professor_id).first().username
#         course_name = course.course_name
    
#         def generate():
#             try:
#                 for chunk in stream(input_text, attachments, history, course_name, professor_name):
#                     yield f"data: {chunk}\n\n"  # SSE 格式
#                     # 在流过程中发送服务状态
#                     yield f"data: {{'status': 'processing', 'chunk': '{chunk}'}}\n\n"
                
#                 full_response = "".join([chunk for chunk in stream(input_text, attachments, history, course_name, professor_name)])
                
#                 # 在此处将完整的响应保存到数据库
#                 chat = Chat.query.filter_by(chat_num=chat_num).first()
#                 if chat:
#                     new_user_message = ChatMessage(
#                         chat_id=chat.chat_id,
#                         role='user',
#                         content=input_text,
#                         branch_id=0
#                     )
#                     db.session.add(new_user_message)
                    
#                     new_assistant_message = ChatMessage(
#                         chat_id=chat.chat_id,
#                         parent_message_id=new_user_message.message_id,
#                         role='assistant',
#                         content=full_response,
#                         branch_id=0
#                     )
#                     db.session.add(new_assistant_message)
                    
#                     chat.last_message_id = new_assistant_message.message_id
#                     db.session.commit()
                    
#                 # 最终状态
#                 yield f"data: {{'status': 'completed', 'response': '{full_response}'}}\n\n"
#             except Exception as e:
#                 yield f"data: {{'status': 'error', 'message': '{str(e)}'}}\n\n"
#                 raise

#         return Response(stream_with_context(generate()), mimetype='text/event-stream')
#     else:
#         return Response(None, mimetype='text/event-stream')