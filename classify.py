from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


MODEL_PATH = 'fashion.model'
LABEL_PATH = 'mlb.pickle'
DATASET_PATH = 'examples/example_01.jpg'


image = cv2.imread(DATASET_PATH)
output = imutils.resize(image, width=400)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


model = load_model(MODEL_PATH)
mlb = pickle.loads(open(LABEL_PATH, "rb").read())

proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

for (i, j) in enumerate(idxs):
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))

cv2.imshow("Output", output)
cv2.waitKey(0)