import requests
import json
import argparse

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--server_url', type=str, help='path of input')
parser.add_argument('--task_id', type=str, help='task id')
parser.add_argument('--input', type=str, help='rtsp url or video path')
args = parser.parse_args()

# url = "http://127.0.0.1:5000/create"
url = args.server_url+"/create"

query = {"video_pt": args.input,
         "task_id": args.task_id}
payload = json.dumps(query)
headers = {
  'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
