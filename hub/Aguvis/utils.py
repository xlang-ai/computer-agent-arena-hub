import base64
import os
import re
import tempfile
import logging
from io import BytesIO
from typing import List, Tuple
from PIL import Image

# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image

def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


# FIXME: hardcoded screen size and planner system message
SCREEN_LOGIC_SIZE = (1280, 720)


def parse_code_from_planner_response(input_string: str) -> List[str]:
    """Parse the planner's response containing executable pyautogui code"""

    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets
    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes


def parse_aguvis_response(input_string, screen_logic_size=SCREEN_LOGIC_SIZE) -> Tuple[str, List[str]]:
    if input_string.lower().startswith("wait"):
        return "WAIT", "WAIT"
    elif input_string.lower().startswith("done"):
        return "DONE", "DONE"
    elif input_string.lower().startswith("fail"):
        return "FAIL", "FAIL"

    try:
        lines = input_string.strip().split("\n")
        lines = [line for line in lines if line.strip() != ""]
        low_level_instruction = lines[0]

        pyautogui_index = -1

        for i, line in enumerate(lines):
            if line.strip() == "assistantos" or line.strip().startswith("pyautogui"):
                pyautogui_index = i
                break

        if pyautogui_index == -1:
            print(f"Error: Could not parse response {input_string}")
            return None, None

        pyautogui_code_relative_coordinates = "\n".join(lines[pyautogui_index:])
        pyautogui_code_relative_coordinates = pyautogui_code_relative_coordinates.replace("assistantos", "").strip()
        corrected_code = correct_pyautogui_arguments(pyautogui_code_relative_coordinates)

        parsed_action = _pyautogui_code_to_absolute_coordinates(corrected_code, screen_logic_size)
        return low_level_instruction, parsed_action
    except Exception as e:
        print(f"Error: Could not parse response {input_string}")
        return None, None

def correct_pyautogui_arguments(code: str) -> str:
    function_corrections = {
        'write': {
            'incorrect_args': ['text', 'content'],
            'correct_args': [],
            'keyword_arg': 'message'
        },
        'press': {
            'incorrect_args': ['key', 'button'],
            'correct_args': [],
            'keyword_arg': None
        },
        'hotkey': {
            'incorrect_args': ['key1', 'key2', 'keys'],
            'correct_args': [],
            'keyword_arg': None
        },
    }

    lines = code.strip().split('\n')
    corrected_lines = []

    for line in lines:
        line = line.strip()
        match = re.match(r'(pyautogui\.(\w+))\((.*)\)', line)
        if match:
            full_func_call = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)

            if func_name in function_corrections:
                func_info = function_corrections[func_name]
                args = split_args(args_str)
                corrected_args = []

                for arg in args:
                    arg = arg.strip()
                    kwarg_match = re.match(r'(\w+)\s*=\s*(.*)', arg)
                    if kwarg_match:
                        arg_name = kwarg_match.group(1)
                        arg_value = kwarg_match.group(2)

                        if arg_name in func_info['incorrect_args']:
                            if func_info['keyword_arg']:
                                corrected_args.append(f"{func_info['keyword_arg']}={arg_value}")
                            else:
                                corrected_args.append(arg_value)
                        else:
                            corrected_args.append(f'{arg_name}={arg_value}')
                    else:
                        corrected_args.append(arg)

                corrected_args_str = ', '.join(corrected_args)
                corrected_line = f'{full_func_call}({corrected_args_str})'
                corrected_lines.append(corrected_line)
            else:
                corrected_lines.append(line)
        else:
            corrected_lines.append(line)

    corrected_code = '\n'.join(corrected_lines)
    return corrected_code

def split_args(args_str: str) -> List[str]:
    args = []
    current_arg = ''
    within_string = False
    string_char = ''
    prev_char = ''
    for char in args_str:
        if char in ['"', "'"]:
            if not within_string:
                within_string = True
                string_char = char
            elif within_string and prev_char != '\\' and char == string_char:
                within_string = False
        if char == ',' and not within_string:
            args.append(current_arg)
            current_arg = ''
        else:
            current_arg += char
        prev_char = char
    if current_arg:
        args.append(current_arg)
    return args

# def extract_coordinates(text, logical_screen_size=SCREEN_LOGIC_SIZE) -> Tuple[int, int] | None:
#     # Pattern to match (x=0.1, y=0.2) or (0.1, 0.2) format
#     text = text.strip()
#     #logger.info(f"Extracting coordinates from: {text}")
#     pattern = r'\((?:x=)?([-+]?\d*\.\d+|\d+)(?:,\s*(?:y=)?([-+]?\d*\.\d+|\d+))?\)'

#     match = re.search(pattern, text)
#     if match:
#         x = int(float(match.group(1)) * logical_screen_size[0])
#         y = int(float(match.group(2)) * logical_screen_size[1]) if match.group(2) else None

#         if y is not None:
#             return (x, y)

#     #logger.info(f"Error: No coordinates found in: {text}")
#     return None


def _pyautogui_code_to_absolute_coordinates(pyautogui_code_relative_coordinates, logical_screen_size=SCREEN_LOGIC_SIZE):
    """
    Convert the relative coordinates in the pyautogui code to absolute coordinates based on the logical screen size.
    """
    import re
    import ast

    width, height = logical_screen_size

    pattern = r'(pyautogui\.\w+\([^\)]*\))'

    matches = re.findall(pattern, pyautogui_code_relative_coordinates)

    new_code = pyautogui_code_relative_coordinates

    for full_call in matches:
        func_name_pattern = r'(pyautogui\.\w+)\((.*)\)'
        func_match = re.match(func_name_pattern, full_call, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            parsed_args = parsed.args
            parsed_keywords = parsed.keywords
        except SyntaxError:
            continue

        function_parameters = {
            'click': ['x', 'y', 'clicks', 'interval', 'button', 'duration', 'pause'],
            'moveTo': ['x', 'y', 'duration', 'tween', 'pause'],
            'moveRel': ['xOffset', 'yOffset', 'duration', 'tween', 'pause'],
            'dragTo': ['x', 'y', 'duration', 'button', 'mouseDownUp', 'pause'],
            'dragRel': ['xOffset', 'yOffset', 'duration', 'button', 'mouseDownUp', 'pause'],
            'doubleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
        }

        func_base_name = func_name.split('.')[-1]

        param_names = function_parameters.get(func_base_name, [])

        args = {}
        for idx, arg in enumerate(parsed_args):
            if idx < len(param_names):
                param_name = param_names[idx]
                arg_value = ast.literal_eval(arg)
                args[param_name] = arg_value

        for kw in parsed_keywords:
            param_name = kw.arg
            arg_value = ast.literal_eval(kw.value)
            args[param_name] = arg_value

        updated = False
        if 'x' in args:
            try:
                x_rel = float(args['x'])
                x_abs = int(round(x_rel * width))
                args['x'] = x_abs
                updated = True
            except ValueError:
                pass
        if 'y' in args:
            try:
                y_rel = float(args['y'])
                y_abs = int(round(y_rel * height))
                args['y'] = y_abs
                updated = True
            except ValueError:
                pass
        if 'xOffset' in args:
            try:
                x_rel = float(args['xOffset'])
                x_abs = int(round(x_rel * width))
                args['xOffset'] = x_abs
                updated = True
            except ValueError:
                pass
        if 'yOffset' in args:
            try:
                y_rel = float(args['yOffset'])
                y_abs = int(round(y_rel * height))
                args['yOffset'] = y_abs
                updated = True
            except ValueError:
                pass

        if updated:
            reconstructed_args = []
            for idx, param_name in enumerate(param_names):
                if param_name in args:
                    arg_value = args[param_name]
                    if isinstance(arg_value, str):
                        arg_repr = f"'{arg_value}'"
                    else:
                        arg_repr = str(arg_value)
                    reconstructed_args.append(arg_repr)
                else:
                    break

            used_params = set(param_names[:len(reconstructed_args)])
            for kw in parsed_keywords:
                if kw.arg not in used_params:
                    arg_value = args[kw.arg]
                    if isinstance(arg_value, str):
                        arg_repr = f"{kw.arg}='{arg_value}'"
                    else:
                        arg_repr = f"{kw.arg}={arg_value}"
                    reconstructed_args.append(arg_repr)

            new_args_str = ', '.join(reconstructed_args)
            new_full_call = f"{func_name}({new_args_str})"
            new_code = new_code.replace(full_call, new_full_call)

    return new_code