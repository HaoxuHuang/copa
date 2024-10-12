import json
import os
import base64
import traceback
import requests
from io import BytesIO

# Get OpenAI API Key from environment variable
api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

metaprompt = '''
- For any marks mentioned in your answer, please highlight them with [].
'''    

chat_history = None

# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_inputs(message, image):

    # # Path to your image
    # image_path = "temp.jpg"
    # # Getting the base64 string
    # base64_image = encode_image(image_path)
    base64_image = encode_image_from_pil(image)
    if chat_history is None:
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "system",
                "content": [
                    metaprompt
                ]
            }, 
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": message, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 800
        }
    else:
        payload = chat_history
        payload['messages'].append({
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": message, 
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        })

    return payload

def request_gpt4v(message, image):
    global chat_history
    payload = prepare_inputs(message, image)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']
    chat_history = payload
    chat_history['messages'].append({
        "role": "assistant",
        "content": res,
    })
    return res

def prepare_inputs_multi_image(messages, images):
    content = []
    for message, image in zip(messages, images):
        content.append({"type": "text", "text": message})
        encode_function = encode_image_from_file if isinstance(image, str) else encode_image_from_pil
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_function(image)}"}})

    if chat_history is None:
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "system",
                "content": [
                    metaprompt
                ]
            }, 
            {
                "role": "user",
                "content": content
            }
            ],
            "max_tokens": 800
        }
    else:
        payload = chat_history
        payload['messages'].append({
            "role": "user",
            "content": content
        })
    
    return payload

def request_gpt4v_multi_image(messages, images):
    global chat_history
    payload = prepare_inputs_multi_image(messages, images)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # print(response.json())
    response = json.loads(response.text)
    try:
        res = response['choices'][0]["message"]["content"]
    except:
        print(response)
        traceback.print_exc()
        return None
    
    chat_history = payload
    chat_history['messages'].append({
        "role": "assistant",
        "content": res,
    })
    return res

def prepare_inputs_multi_image_behavior(messages, images):
    content = []
    for message, image in zip(messages, images):
        content.append({"type": "text", "text": message})
        encode_function = encode_image_from_file if isinstance(image, str) else encode_image_from_pil
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_function(image)}"}})

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "system",
            "content": [
                'Your response must strictly adhere to the specified format.'
            ]
        }, 
        {
            "role": "user",
            "content": content
        }
        ],
        "max_tokens": 800
    }
    
    return payload

def request_gpt4v_multi_image_behavior(messages, images):
    payload = prepare_inputs_multi_image_behavior(messages, images)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']
    return res
