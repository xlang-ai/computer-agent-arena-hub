"""
Utility functions for OpenAI CUA agent - directly adapted from openai-cua-quickstart
"""
import os
import json
import requests
import time
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Any, Union

def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response == "y"


def sanitize_message(message):
    """Sanitize a message to be sent to the API"""
    if isinstance(message, dict):
        if message.get("role") and message.get("content"):
            return message
    return {"role": "user", "content": str(message)}

def create_response(**kwargs):
    """Create a response from the OpenAI API - directly from openai-cua-quickstart"""
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
        "Openai-beta": "responses=v1",
    }

    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["Openai-Organization"] = openai_org

    response = requests.post(url, headers=headers, json=kwargs)

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")

    return response.json()

def pp(obj):
    """Print an object in a pretty format"""
    print(json.dumps(obj, indent=2))

def show_image(base64_str, format="JPEG"):
    """Display an image from a base64 string"""
    img = Image.open(BytesIO(base64.b64decode(base64_str)))
    return img

def encode_image_to_base64(image_data):
    """Convert PIL Image to base64 string"""
    if isinstance(image_data, Image.Image):
        buffered = BytesIO()
        image_data.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    return image_data  # Assume it's already a base64 string 