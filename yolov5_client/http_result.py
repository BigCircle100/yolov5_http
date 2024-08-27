import requests
import json
import os
import base64
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--server_url', type=str, help='path of input')
parser.add_argument('--task_id', type=str, help='task id')
args = parser.parse_args()

# url = "http://127.0.0.1:5000/result"
url = args.server_url+"/result"

query = {"task_id": args.task_id}
payload = json.dumps(query)
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
if response.status_code != 200:
    print("result query error")
else:
    results = response.json()

    save_dir = './img_with_results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for result in results:
        # frame_id = result.get('frame_id')
        task_id = query['task_id']
        jpg_base64 = result.get('jpg_base64')
        current_time = datetime.now()
        formatted_time = current_time.strftime("%m-%d-%H-%M-%S-%f")
        if jpg_base64:
            image_data = base64.b64decode(jpg_base64)
            save_path = os.path.join(save_dir, f'{task_id}_{formatted_time}.jpg')
            with open(save_path, 'wb') as f:
                f.write(image_data)
