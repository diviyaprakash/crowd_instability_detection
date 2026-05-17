import requests
import time

try:
    print("Testing /upload")
    files = {'video': open('test.mp4', 'wb')}  # mock file
    r = requests.post("http://127.0.0.1:5000/upload", files=files)
    print("/upload response:", r.status_code, r.text)

    print("Testing /stop")
    r = requests.post("http://127.0.0.1:5000/stop")
    print("/stop response:", r.status_code, r.text)

    print("Testing /upload again")
    files = {'video': open('test.mp4', 'rb')}  # mock file
    r = requests.post("http://127.0.0.1:5000/upload", files=files)
    print("/upload response:", r.status_code, r.text)

except Exception as e:
    print(e)
