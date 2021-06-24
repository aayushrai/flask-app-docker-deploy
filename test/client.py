import requests
import cv2
import base64
import json

# reading image
img = cv2.imread("pawan.jpeg")
retval, buffer = cv2.imencode('t.jpg', img)
# Convert to base64 encoding and show start of data
jpg_as_text = base64.b64encode(buffer)
# gUrl = "https://face-recog-flask-server-deploy-xfzfneltta-df.a.run.app"
gUrl = "http://localhost:5000"
# registring face inside image with id pawan
url = gUrl + "/registration"
x = requests.post(url, data = {"image":jpg_as_text,"id":"aayush"})
print(x.text)


# face recognition of face we just register above
url = gUrl + "/facerecog"
x = requests.post(url, data = {"image":jpg_as_text})
print(x.text)

url = gUrl
x = requests.get(url)
print(x.text)
