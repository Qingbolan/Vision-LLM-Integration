import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv('../../.env')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE') or None)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://chat.scholarhero.cn/api/FOKn/UserRegion/2402-CS460D1-78267/Prof.Silan Hu/temp/raw_092bdc7a-f226-4bd9-a192-f9cc073cc448.png",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])