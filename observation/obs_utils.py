"""
Utility functions for processing observations
"""

import base64
import tiktoken
from typing import Dict
import xml.etree.ElementTree as ET

from .a11y_tree_utils import (
    draw_bounding_boxes, 
    linearize_accessibility_tree,
    filter_nodes
    )

def process_screenshot( obs: Dict) -> str:
    """Process screenshot observation."""
    if not obs.get('screenshot'):
        return ''
    return encode_image(obs['screenshot'])

def process_a11y_tree( obs: Dict, platform: str) -> str:
    """Process accessibility tree observation."""
    if not obs.get('a11y_tree'):
        return ''
    tree = linearize_accessibility_tree(
        accessibility_tree=obs['a11y_tree'],
        platform=platform
    )
    return trim_accessibility_tree(tree)

def process_som( obs: Dict, platform: str) -> str:
    """Process SOM (Screen Object Model) observation."""
    if not (obs.get('screenshot') and obs.get('a11y_tree')):
        return ''
    _mask, drew_nodes, tagged_screenshot, _ = tag_screenshot(
        obs['screenshot'], 
        obs['a11y_tree'], 
        platform
    )
    return encode_image(tagged_screenshot)

def encode_image(image_content):
    """Encode the image content to a base64 string.

    Args:
        image_content: The content of the image

    Returns:
        str: The base64 encoded string
    """
    return base64.b64encode(image_content).decode('utf-8')


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens=10000):
    """Trim the linearized accessibility tree to a certain number of tokens.

    Args:
        linearized_accessibility_tree: The linearized accessibility tree
        max_tokens: The maximum number of tokens

    Returns:
        str: The trimmed linearized accessibility tree
    """
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


def tag_screenshot(screenshot, accessibility_tree, platform="Ubuntu"):
    """Tag the screenshot with the accessibility tree.

    Args:
        screenshot: The screenshot
        accessibility_tree: The accessibility tree
        platform: The platform of the accessibility tree

    Returns:
        marks: The marks
        drew_nodes: The drew nodes
        tagged_screenshot: The tagged screenshot
        element_list: The element list
    """
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    marks, drew_nodes, tagged_screenshot, element_list = draw_bounding_boxes(
        nodes, screenshot)

    return marks, drew_nodes, tagged_screenshot, element_list

