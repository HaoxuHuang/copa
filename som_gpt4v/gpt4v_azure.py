import requests
import json
import os
import base64
import traceback
from io import BytesIO

api_base = ...
deployment_name = ...
API_KEY = os.environ.get('AZURE_API_KEY')

base_url = f"{api_base}openai/deployments/{deployment_name}" 
headers = {   
    "Content-Type": "application/json",   
    "api-key": API_KEY 
}
endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview"

def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('ascii')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')

def encode_image_base64(img):
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('ascii')

chat_history = None

def prepare_inputs_multi_image(messages, images):
    content = []
    for message, image in zip(messages, images):
        content.append({"type": "text", "text": message})
        encode_function = encode_image_from_file if isinstance(image, str) else encode_image_from_pil
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_function(image)}"}})

    if chat_history is None:
        payload = {
            "messages": [
            {
                "role": "system",
                "content": '- For any marks mentioned in your answer, please highlight them with [].'
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

def request_gpt4v_multi_image_azure(messages, images):
    global chat_history
    payload = prepare_inputs_multi_image(messages, images)
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
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
        "messages": [
        {
            "role": "system",
            "content": 'Your response must strictly adhere to the specified format.'
        }, 
        {
            "role": "user",
            "content": content
        }
        ],
        "max_tokens": 800
    }
    
    return payload

def request_gpt4v_multi_image_behavior_azure(messages, images):
    payload = prepare_inputs_multi_image_behavior(messages, images)
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
    response = json.loads(response.text)
    res = response['choices'][0]["message"]["content"]
    return res