"""
Utility functions for the Prompt agent.
"""
import base64
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import re

def decode_image_from_base64(base64_string):
    """Decode an image from a base64 string.

    Args:
        base64_string: The base64 string
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def encode_image_to_base64(image):
    """Encode an image to a base64 string.

    Args:
        image: The image
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def draw_grid_with_number_labels(image, grid_size=100):
    """Draw a grid with number labels on an image.

    Args:
        image: The image
        grid_size: The grid size
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)  # Use a system font if available
    except IOError:
        font = ImageFont.load_default()
    
    # Draw vertical lines and label x-axis numbers
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill="red", width=1)
        draw.text((x + 5, 5), str(x), fill="red", font=font)  # Top label for x-axis

    # Draw horizontal lines and label y-axis numbers
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill="red", width=1)
        draw.text((5, y + 5), str(y), fill="red", font=font)  # Left label for y-axis

    return image


def parse_actions_from_string(input_string):
    """Parse the actions from the response.

    Args:
        input_string: The input string
    """
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r'```json\s+(.*?)\s+```', input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r'```\s+(.*?)\s+```', input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)

def parse_code_from_string(input_string):
    """Parse the code from the response.

    Args:
        input_string: The input string
    """
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
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes

def parse_code_from_som_string(input_string, masks):
    """Parse the code from the response.

    Args:
        input_string: The input string
        masks: The masks
    """
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += "tag_" + str(i + 1) + "=" + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ['WAIT', 'DONE', 'FAIL']:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions
