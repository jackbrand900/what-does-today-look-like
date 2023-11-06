import requests

class ImageGen:
    def __init__(self) -> None:
        self.api_url = 'https://api-inference.huggingface.co/models/cloudqi/cqi_text_to_image_pt_v0'
        self.headers = self.load_headers()
    
    def load_headers(self): 
        key=open('hf_key.txt', 'r').read()
        headers = {'Authorization': 'Bearer ' + key}
        return headers

    def query(self, caption):
        payload = {'inputs': caption}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.content
    
