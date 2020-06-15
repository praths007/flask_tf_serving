import base64
import json
from io import BytesIO
from tensorflow.keras.models import load_model
import numpy as np
import requests
from flask import Flask, request

# from flask_cors import CORS

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


# prediction URL
@app.route('/mnist_predict/', methods=['POST'])
def mnist_predict():
    model = load_model("mnist/1")

    ip_data = request.json

    for_prediction = np.array(ip_data["values"])
    # the ip array to predict should be (1,28,28,1) but since we decode a json string which has an extra dimension
    # which is the key out of the key value pair it needs to be removed using 0
    prediction = model.predict_classes(for_prediction[0, :, :, :])

    return json.dumps(prediction.tolist())


if __name__ == "__main__":
    app.run()
