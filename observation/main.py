from typing import List, Dict, Iterator
import logging
from ..utils.schemas import ObservationType
from .obs_utils import process_screenshot, process_a11y_tree, process_som

class Observation:
    def __init__(self, obs_options: List[str]):
        self._obs_options = None
        self.obs_options = obs_options
    
    @property
    def obs_options(self) -> List[str]:
        return self._obs_options
    
    @obs_options.setter
    def obs_options(self, value):
        if not isinstance(value, list):
            raise TypeError("obs_options must be a list")
        
        if not value:
            raise ValueError("obs_options must contain at least one observation type")
        
        input_set = set(value)
        valid_types = {t.value for t in ObservationType}
        
        invalid_types = input_set - valid_types
        if invalid_types:
            raise ValueError(f"Invalid observation types: {invalid_types}")
        
        logging.info(f"Observation initialized with options: {value}")
        self._obs_options = value
    
    def __iter__(self) -> Iterator[tuple]:
        for obs_type in self.obs_options:
            yield (obs_type, True)
    
    def to_dict(self) -> Dict[str, bool]:
        return dict(self)

    def to_string(self) -> str:
        return ",".join(self.obs_options)
    
    def to_tuple(self) -> tuple:
        return tuple(self.obs_options)
    
    def process_observation(self, raw_obs: Dict, platform: str) -> Dict:
        """Process raw observations into a standardized dictionary format.
        
        Args:
            obs: Raw observation dictionary containing various observation types
            
        Returns:
            Processed observation dictionary with encoded/formatted values
        """
        agent_obs_dict = {}
        
        # Define processors for each observation type
        processors = {
            ObservationType.SCREENSHOT.value: process_screenshot,
            ObservationType.A11Y_TREE.value: lambda x: process_a11y_tree(x, platform),
            ObservationType.TERMINAL.value: lambda x: x.get('terminal', '') if x else '',
            ObservationType.SOM.value: lambda x: process_som(x, platform),
            ObservationType.HTML.value: lambda x: x.get('html', '') if x else ''  # Simple passthrough for now
        }

        # Process each observation type that exists in input
        for obs_type, processor in processors.items():
            if obs_type in raw_obs:
                agent_obs_dict[obs_type] = processor(raw_obs)

        return agent_obs_dict

