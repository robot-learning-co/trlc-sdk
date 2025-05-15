import requests
from PIL import Image
import numpy as np

class HTTPClient:
    def __init__(self):
        self.base_url = "http://api.robot-learning.co"

    def run_segmentation(self, files: dict, data: dict) -> str:
        url = f"{self.base_url}/segmentation"
        
        response = requests.post(
            url,
            files=files,
            data=data
        )
        if response.status_code != 200:
            raise Exception(f"Failed to run pose estimation: {response.status_code}")
        
        with Image.open(files["image"]) as img_pil:
            w, h = img_pil.size
        
        return np.array(response.json()["mask"]).reshape(h, w)
    
    
    def run_pose_estimation(self, files: dict, data: dict) -> str:
        url = f"{self.base_url}/pose_estimation"
        
        response = requests.post(
            url,
            files=files,
            data=data
        )
        if response.status_code != 200:
            raise Exception(f"Failed to run pose estimation: {response.status_code}")
        
        return response.json()["pose"]