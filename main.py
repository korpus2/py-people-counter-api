import os.path
import ssl
import urllib.request

import cv2
import numpy as np
from flask import Flask
from flask_restful import Resource, Api, request
from werkzeug.utils import secure_filename

ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

TMP_DIR = "/Users/krzysztofkorus/Desktop/Studia/Cloudy/py-people-counter-api/"


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread("images/dworzecWaw.jpeg")
        print(type(img))
        print(img.shape)
        boxes, weights = hog.detectMultiScale(img, winStride=(2, 2))
        return {"count": len(boxes)}


# Example: /web/?url=
class PeopleCounterParams(Resource):
    def get(self):
        path = request.args["url"]
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        boxes, weights = hog.detectMultiScale(img, winStride=(2, 2))
        return {"count": len(boxes)}


class PeopleCounterUpload(Resource):
    def post(self):
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(TMP_DIR, filename)
        file.save(filepath)
        img = cv2.imread(filepath)
        boxes, weights = hog.detectMultiScale(img, winStride=(2, 2))
        os.remove(filepath)
        return {"count": len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}


api.add_resource(PeopleCounter, "/")
api.add_resource(PeopleCounterParams, "/web/")
api.add_resource(PeopleCounterUpload, "/upload/")
api.add_resource(HelloWorld, "/test")


if __name__ == "__main__":
    app.run(debug=True, port=2137)
