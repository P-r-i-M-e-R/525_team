
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import time

from tqdm import tqdm

api_key = os.environ['api_key']

def call_api(url, data):
    headers = { "Authorization" : f"Api-Key {api_key}" }
    return requests.post(url, json=data, headers=headers).json()

def call_api_get(url, data):
    headers = { "Authorization" : f"Api-Key {api_key}" }
    return requests.get(url, headers=headers).json()


def get_ocr(images_list: list[Image]) -> list[str]:
    
    json_list = []
    for img in tqdm(images_list, desc="Images"):
        buffer = BytesIO()
        img.save(buffer,format="JPEG")
        myimage = buffer.getvalue()
        img = base64.b64encode(myimage).decode('utf-8')
        j = {
          "mimeType": "JPEG",
          "languageCodes": ["ru"],
          "model": "page",
          "content": img
        }
        json_list.append(j)
    
    results = []
    for idx,img in tqdm(enumerate(json_list), total=len(json_list), desc="Making OCR requests"):
        for _ in range(10):
            try:
                res = call_api("https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText",img)
                results.append(res['result']['textAnnotation']['fullText'])
                break
            except: 
                time.sleep(0.1)  
                
    return results


def correct_with_llm(ocr_texts, api_key, folder_id) -> '[corrected texts]':
    from yandex_cloud_ml_sdk import YCloudML
    folder_id = os.environ['folder_id']

    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    yandexgpt = sdk.models.completions("yandexgpt", model_version="rc")
    cleaned_txts = []

    prompt = """
    Ты - газетный редактор, которому на вход подается распознанный с помощью OCR текст газетной полосы.
    Твоя задача:
    * вернуть исходный текст с минимальными изменениями
    * исправить в этом тексте опечатки
    """

    for item in ocr_texts:
        if item:
            cleaned = yandexgpt.run([
                {
                    "role": "system",
                    "text": prompt
                },
                {
                    "role": "user",
                    "text": item
                }])
            time.sleep(0.1)
            cleaned_txts.append(cleaned.text)
        else:
            cleaned_txts.append('')

    return (cleaned_txts)