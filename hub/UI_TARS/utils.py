"""
Utility functions for UI-TARS agent.
Parse model outputs to actions and PyAutoGUI code.
"""
import re
from copy import deepcopy
import ast
import random
import base64
import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, List, Union, Optional, Any

# Constants for action types
FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

def decode_image_from_base64(base64_string: str) -> np.ndarray:
    """Decode base64 string to image."""
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode image to base64 string."""
    try:
        _, img_encoded = cv2.imencode('.png', image)
        base64_string = base64.b64encode(img_encoded).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def draw_grid_with_number_labels(image: np.ndarray, grid_size: int = 100) -> np.ndarray:
    """Draw grid with number labels on image."""
    img = image.copy()
    height, width = img.shape[:2]
    
    # Draw vertical lines
    for x in range(0, width, grid_size):
        cv2.line(img, (x, 0), (x, height), (128, 128, 128), 1)
        
    # Draw horizontal lines
    for y in range(0, height, grid_size):
        cv2.line(img, (0, y), (width, y), (128, 128, 128), 1)
        
    # Add number labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            label = f"({x},{y})"
            cv2.putText(img, label, (x+5, y+15), font, font_scale, (255, 0, 0), 1)
            
    return img

def new_action_to_old_action(action: dict, image_width: int, image_height: int) -> dict:
    """Convert new action format to old format."""
    old_format_action = {
        "type": action["type"],
        "custom": {},
        "boxes": [],
    }
    if "start_box" in action["params"]:
        start_box = deepcopy(action["params"]["start_box"])
        start_box = eval(start_box)
        if len(start_box) == 2:
            start_box = start_box + start_box
        start_box[0] *= image_width
        start_box[1] *= image_height
        start_box[2] *= image_width
        start_box[3] *= image_height
        old_format_action["boxes"].append(start_box)
    if "end_box" in action["params"]:
        end_box = deepcopy(action["params"]["end_box"])
        end_box = eval(end_box)
        if len(end_box) == 2:
            end_box = end_box + end_box
        end_box[0] *= image_width
        end_box[1] *= image_height
        end_box[2] *= image_width
        end_box[3] *= image_height
        old_format_action["boxes"].append(end_box)
    for key, value in action["params"].items():
        if key in ["type", "start_box", "end_box"]: continue
        old_format_action["custom"][key] = value
    return old_format_action

def parse_action(action_str: str) -> dict:
    """Parse action string to dictionary."""
    try:
        node = ast.parse(action_str, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }
    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text: str) -> str:
    """Escape single quotes in text."""
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

# ... [Previous functions continued]

def parse_action_qwen2vl(text: str, factor: int, image_height: int, image_width: int) -> List[Dict]:
    """Parse QWen-VL model output to action list.
    
    Args:
        text: Model output text
        factor: Scaling factor for coordinates
        image_height: Height of the image
        image_width: Width of the image
        
    Returns:
        List of parsed actions with their parameters
    """
    text = text.strip()
    
    # Determine thought pattern based on text start
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
        
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
            
    if "Action:" not in text:
        raise ValueError("No 'Action:' found in text")
    action_str = text.split("Action:")[-1]

    # Split multiple actions
    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if not action_str.strip():
            continue
            
        if "type(content" in action_str:
            # Handle special case for type actions
            def escape_quotes(match):
                content = match.group(1)
                return content

            pattern = r"type\(content='(.*?)'\)"
            content = re.sub(pattern, escape_quotes, action_str)
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    # Parse all actions
    parsed_actions = []
    for action in all_action:
        if not action.strip():
            continue
        parsed = parse_action(action.replace("\n","\\n").lstrip())
        if parsed:
            parsed_actions.append(parsed)
        else:
            print(f"Warning: Failed to parse action: {action}")

    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            print(f"Action can't parse: {raw_str}")
            continue
            
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if not param:
                continue
                
            param = param.lstrip()
            action_inputs[param_name.strip()] = param
            
            # Handle coordinate boxes
            if "start_box" in param_name or "end_box" in param_name:
                try:
                    ori_box = param
                    numbers = ori_box.replace("(", "").replace(")", "").split(",")
                    float_numbers = [float(num.strip()) / factor for num in numbers]
                    
                    if len(float_numbers) == 2:
                        float_numbers = float_numbers * 2
                    elif len(float_numbers) != 4:
                        raise ValueError(f"Invalid box format: {ori_box}")
                        
                    action_inputs[param_name.strip()] = str(float_numbers)
                except Exception as e:
                    print(f"Error processing box coordinates: {e}")
                    continue

        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
        
    return actions

def parse_refine_coordinate_response(text: str, factor: int = 1000) -> Dict[str, str]:
    """Parse coordinate refinement response.
    
    Args:
        text: Response text containing coordinates
        factor: Scaling factor for coordinates
        
    Returns:
        Dictionary mapping box types to coordinate strings
    """
    pattern = r"(start_box|end_box)='?\((\d+),(\d+)\)'?"
    matches = re.findall(pattern, text)
    results = {}
    
    if matches:
        for match in matches:
            try:
                key = match[0]
                x = int(match[1])
                y = int(match[2])
                coordinates = (x / factor, y / factor, x / factor, y / factor)
                results[key] = str(list(coordinates))
            except (ValueError, IndexError) as e:
                print(f"Error parsing coordinates: {e}")
                continue
    
    return results

def parsing_response_to_pyautogui_code(responses: Union[Dict, List], 
                                     image_height: int, 
                                     image_width: int, 
                                     input_swap: bool = True) -> str:
    """Convert model response to PyAutoGUI code.
    
    Args:
        responses: Model response dictionary or list of responses
        image_height: Height of the screen
        image_width: Width of the screen
        input_swap: Whether to use clipboard for text input
        
    Returns:
        Generated PyAutoGUI code string
    """
    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(3)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
        
        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{content.strip()}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{content.strip()}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code

if __name__ == '__main__':
    # Example usage and testing
    def run_tests():
        # Test action parsing
        mock_response = """Thought: Testing action parsing
Action: finished(content='(873,667)')"""
        mock_response = mock_response.replace("Thought:", "Action_Summary:")
        
        try:
            mock_response_dict = parse_action_qwen2vl(mock_response, 1000, 720, 1080)
            print("Parsed action:", mock_response_dict)
            
            rc_response = parse_refine_coordinate_response(
                "drag(start_box='(579,853)', end_box='(607,853)')"
            )
            print("Refined coordinates:", rc_response)
            
            response_dict = parsing_response_to_pyautogui_code(
                mock_response_dict, 720, 1080
            )
            print("Generated PyAutoGUI code:", response_dict)
            
        except Exception as e:
            print(f"Test failed: {e}")
            
    run_tests() 