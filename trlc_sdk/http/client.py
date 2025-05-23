import requests
import time
import json
from trlc_sdk.utils.network import rle_to_single_mask

def submit_request(url, payload, verbose):
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to submit request: {response.status_code}")
    call_id = response.json()["call_id"]
    if verbose:
        print(f"Spawned job with call_id: {call_id}")
    return call_id

def poll_results(url, call_id, timeout=60, verbose=False):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if verbose:
            print(f"Polling results for call_id: {call_id}")
        response = requests.get(url + f"/{call_id}")
        if response.status_code == 200:
            if verbose:
                print(f"Received results for call_id: {call_id}")
            return response.json()
        if response.status_code == 404:
            raise Exception(response.json())
        
        time.sleep(1)
    
    raise Exception("Timeout waiting for results")


class HTTPClient:
    def __init__(self):
        self.base_url = "https://api.robot-learning.co/v0"
        
    def segment(self, image_base64: str, text_prompt: str = None, verbose=False):
        
        payload = {"image_base64": image_base64, "text_prompt": text_prompt}
        call_id = submit_request(url=f"{self.base_url}/segmentation", payload=payload, verbose=verbose)
        results = poll_results(url=f"{self.base_url}/segmentation", call_id=call_id, verbose=verbose)
        
        results = results["results"]
                
        for ann in results["annotations"]:
            ann["segmentation"] = rle_to_single_mask(ann["segmentation"])
            
        return results
    
    def estimate_pose(self, 
                      rgb_base64: str,
                      depth_base64: str,
                      mask_base64: str,
                      cam_K: list[list[float]],
                      mesh_obj: str,
                      mesh_mtl: str,
                      mesh_png: str,
                      verbose=False):
        
        payload = {
            "rgb": rgb_base64,
            "depth": depth_base64,
            "mask": mask_base64,
            "cam_K": cam_K,
            "mesh_obj": mesh_obj,
            "mesh_mtl": mesh_mtl,
            "mesh_png": mesh_png
        }
        
        call_id = submit_request(url=f"{self.base_url}/pose-estimation", payload=payload, verbose=verbose)
        results = poll_results(url=f"{self.base_url}/pose-estimation", call_id=call_id, verbose=verbose)
        
        return results
