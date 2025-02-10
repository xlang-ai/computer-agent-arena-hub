from typing import Callable, Any, Optional, Tuple
import base64


class DesktopEnv:
    def __init__(
        self,
        action_space: str = "computer_13",
        screen_size: Tuple[int] = (1920, 1080),
        **kwags,
        ):
        self.obs_options = {}  
        self._step_no = 0
        self.action_history = []
        self.action_space = action_space
        self.resolution = screen_size
        
        self.screenshots = [
            self._load_image("test_env/test_observations/screenshot0.jpg"),
            self._load_image("test_env/test_observations/screenshot1.jpg"),
        ]
        
        self.a11y_trees = [
            self._load_accessibility_tree("test_env/test_observations/a11y_tree0.txt"),
            self._load_accessibility_tree("test_env/test_observations/a11y_tree1.txt"),
        ]
    
    def _get_screenshot(self):
        if self._step_no == 0:
            return self.screenshots[0]
        return self.screenshots[1]
    
    def _get_accessibility_tree(self):
        if self._step_no == 0:
            return self.a11y_trees[0]
        return self.a11y_trees[1]

    def set_obs_options(self, obs_options):
        print(f"Setting obs options to {obs_options}")
        self.obs_options = obs_options
    
    def _load_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                # Read the image file in binary mode
                image_data = image_file.read()
                # Encode the binary data as Base64
                return image_data
        except FileNotFoundError:
            print(f"Error: File not found at {image_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def _load_accessibility_tree(self, tree_path):
        try:
            with open(tree_path, "r") as tree_file:
                # Read the accessibility tree file
                tree_data = tree_file.read()
                return tree_data
        except FileNotFoundError:
            print(f"Error: File not found at {tree_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_obs(self):
        obs = {}
        obs["screenshot"] = self._get_screenshot()
        if 'a11y_tree' in self.obs_options and self.obs_options['a11y_tree']:
            obs["a11y_tree"] = self._get_accessibility_tree()
        if 'terminal' in self.obs_options and self.obs_options['terminal']:
            obs["terminal"] = ""
        if 'instruction' in self.obs_options and self.obs_options['instruction']:
            obs["instruction"] = "Open Chrome browser"
        
        return obs
    
    def _start_video_recording(self):
        pass
    
    def _stop_video_recording(self):
        pass
    
    def step(self, action) -> Tuple:
        self._step_no += 1
        self.action_history.append(action)

        info = {}
        terminated = False  # todo: Define episode termination condition for each example
        
        if action == 'FAIL' or action == 'DONE':
            terminated = True
            
        else:       
            if self.action_space == "claude_computer_use":
                tool_result = {
                    "role": "user",
                    "content": [
                        {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                        "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": self.screenshots[1],
                                    }
                                }
                            ]
                        }
                    ]
                }
                info.update({"tool_result": tool_result})
        
        return (terminated, info)