import json
import numpy as np
import requests

import pickle

X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

# this is necessary to send json as string because np array cannot be sent as is
# this method can be used for any np array
# convert to tolist to convert array to text
scoring_payload_custom = {"values": [X_test[0:1].tolist()]}

# sending post request to custom Flask server
# r = requests.post('http://localhost:5000/mnist_predict/', json=scoring_payload_custom)

# print(json.loads(r.content.decode()))
# print(np.argmax(y_test[0]))

######################################################
# sending post request to TensorFlow Serving
scoring_payload_tf_serving = {
    "instances": [X_test[0:1].tolist()][0]
}

r = requests.post('http://localhost:8501/v1/models/mnist:predict', json=scoring_payload_tf_serving)

predictions = json.loads(r.content.decode())
prediction = np.argmax(predictions["predictions"])

print(prediction)
print(np.argmax(y_test[0]))
