import json
import os
import re
import jwt
import pytz
import uuid
import subprocess
import base64
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime
from typing import Optional, Union, Any, List, Dict

from models.BackboneModel import BackboneModel
from test_env.logger import agent_logger as logger


    
def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_all_click_coordinates(action):
    coords = []

    if isinstance(action, str):
        # Extract coordinates for pyautogui.click and pyautogui.moveTo
        click_matches = re.findall(r"pyautogui\.click\s*\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", action)
        move_matches = re.findall(r"pyautogui\.moveTo\s*\(\s*(\d+)\s*,\s*(\d+)", action)
        
        # Convert matched coordinates to integer tuples and combine lists
        click_coords = [(int(x), int(y)) for x, y in click_matches]
        move_coords = [(int(x), int(y)) for x, y in move_matches]
        coords = click_coords + move_coords
        
    elif isinstance(action, list):
        if not action or not isinstance(action[0], dict):
            return coords
        
        for action_item in action:
            if action_item.get('name') == 'computer' and action_item.get('input',{}).get('coordinate'):
                coords.append(action_item['input']['coordinate'])
                
    # Return combined coordinates
    return coords

def draw_multiple_clicks_on_image(image, coordinates_list, color="red", size=25, width=4):
    draw = ImageDraw.Draw(image)
    for x, y in coordinates_list:
        left = x - size
        top = y - size
        right = x + size
        bottom = y + size
        draw.rectangle((left, top, right, bottom), outline=color, width=width)

def process_action_and_visualize_multiple_clicks(action, screenshot_base64):
    if not isinstance(action, str) or not screenshot_base64:
        return screenshot_base64
    
    click_coords_list = parse_all_click_coordinates(action)
    image = decode_base64_image(screenshot_base64)
    if click_coords_list:
        draw_multiple_clicks_on_image(image, click_coords_list)
    return encode_image_to_base64(image) 

helper_model = BackboneModel(model_name='gpt-4o-mini-2024-07-18')

def simplify_action(action) -> str:
    try:
        if isinstance(action, dict) or isinstance(action, list):
            action = json.dumps(action)
        
        messages = [
            {
                "role": "system",
                "content": """
You are an action simplification assistant. Your task is to convert various types of action inputs into simple, human-readable descriptions.

For different input types:

1. For Python code (especially pyautogui commands):
- Format as `Action: Parameter`
- Examples:
  * `Click: (x, y)`
  * `Type: Text`
  * `Drag: from (x, y) to (x, y)`
  * `Hotkey: X`
  * Simple `Click` if no parameters
  * If there is moveTo and click, output 'Move to (x, y) and Click'.
  * Do not output 'Wait' or 'Sleep'

2. For JSON/dict-style tool actions:
- Extract the core action and parameters
- Try to follow the format above
- Examples:
  * From `{'action': 'screenshot', 'type': 'computer'}` → `Take Screenshot`
  * From `{'tool': 'browser', 'action': 'navigate', 'url': 'example.com'}` → `Navigate to: example.com`

3. For natural language or other formats:
- Provide a concise summary of the action
- Keep important parameters or targets
- Make it clear and readable

General rules:
- Keep descriptions short and clear
- Focus on the main action and essential parameters
- Separate multiple actions with newlines
- If input is unclear or invalid, return it unchanged
""".strip()
            },
            {
                "role": "user",
                "content": f"Please simplify the following action:\n\n{action}"
            }
        ]

        # Call the helper model to get the completion
        response = helper_model.completion(messages=messages)
        
        # Check and extract the response text if the response is valid
        if response and "response_text" in response:
            simplified_action_text = response["response_text"]
            
        else:
            simplified_action_text =  action
            
        return simplified_action_text
    
    except Exception as e:
        return action
    
def get_temp_video_url(video_path):
    pass

def pretty_print(text):
    pass