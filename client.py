import requests
import cv2
import base64
import json

# reading image
img = cv2.imread("aayush.jpg")
retval, buffer = cv2.imencode('aayush.jpg', img)
# Convert to base64 encoding and show start of data
jpg_as_text = base64.b64encode(buffer)


# registring face inside image with id 12321
# url = "http://0.0.0.0:5000/registration"
# x = requests.post(url, data = {"image":jpg_as_text,"id":"12321"})
# print(x.text)


# face recognition of face we just register above
url = "http://0.0.0.0:5000/facerecog"
x = requests.post(url, data = {"image":jpg_as_text})
print(x.text)

