import re
import json

def extract_image_urls(data):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    image_urls = []
    
    for item in data:
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

# 示例数据
attachments = '''
[{"file_id":"","file_name":"DALL·E 2023-10-15 00.43.56","file_type":"image/png","file_url":"http://chat.scholarhero.cn/api/FOKn/UserRegion/test000/Prof.Silan_Hu/temp/raw_c2d5c473-0d0f-467a-ae90-03156a2ae7a5.png"}]
'''

# 将 attachments 转换为 Python 字典
data = json.loads(attachments)

result = extract_image_urls(data)
print("----")
print(result)
print(type(result))

print(result[0]["image_url"]["url"])
# for item in result:
#     print(item)
#     for key, value in item.items():
#         print(key)
#         print(value)
print("----")