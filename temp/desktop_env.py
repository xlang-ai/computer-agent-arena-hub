class DesktopEnv:
    def __init__(self):
        self.obs_options = {}  
    
    def get_screenshot(self):
        pass

    def set_obs_options(self, obs_options):
        print(f"Setting obs options to {obs_options}")
        self.obs_options = obs_options