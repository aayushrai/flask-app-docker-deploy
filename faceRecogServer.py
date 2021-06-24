from flask import Flask, request
import cv2
import base64
import numpy as np
from Resoluteai.ResoluteaiFaceRecog import ResoluteaiFaceRecog
import os

app = Flask(__name__)

#define paths
BASE_DIR = "assets"
MODEL_DIR = os.path.join(BASE_DIR,"models/knownUsersEncoding")
UNKNOWN_DIR =  os.path.join(BASE_DIR,"users/unknownUsers")
UNKNOWN_MODEL_DIR =  os.path.join(BASE_DIR,"models/unknownUsersEncoding")
USER_PIC_DIR = os.path.join(BASE_DIR,"users/knownUsers")

faceEngine = ResoluteaiFaceRecog(BASE_DIR,MODEL_DIR,UNKNOWN_DIR,UNKNOWN_MODEL_DIR,USER_PIC_DIR)

@app.route('/')
def hello_world():
    return 'Hello, Welcome to resolute face recog!'


@app.route("/facerecog", methods=["POST"])
def facerecog():
    global faceEngine
    img = request.form.get("image")
    jpg_original = base64.b64decode(img)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv2.imdecode(jpg_as_np, flags=1)
    frame,detected_users_list,area = faceEngine.det_recog_engine(image_buffer)
    print(detected_users_list)
    return detected_users_list[0]


@app.route("/registration", methods=["POST"])
def resigter():
    global faceEngine
    img = request.form.get("image")
    userId = request.form.get("id")
    jpg_original = base64.b64decode(img)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv2.imdecode(jpg_as_np, flags=1)
    path = os.path.join(USER_PIC_DIR,str(userId))
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path+"/1.jpg",image_buffer)
    faceEngine.retrain_fn(new_users=[userId])
    return 'Successfully train!'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT",5000)))