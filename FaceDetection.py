import cv2 as cv
import os
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

xml = "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(xml)
model = load_model("mask_prediction")
cap = cv.VideoCapture(0)

image_w = 64
image_h = 64

data_list = []

while (True):
    ret, frame = cap.read()
    frame = cv.flip(frame,1)
    faces = face_cascade.detectMultiScale(frame)

    for (x,y,w,h) in faces :
        face = frame[y:y+h, x:x+w]
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        face = face.resize((image_w, image_h))
        face = np.asarray(face)
        data_list.append(face)
        """tf.convert_to_tensor(
            data_list, dtype=None, dtype_hint=None, name=None
        )
        predictions = model.predict(data_list)
        np.argmax(predictions[0])
    """

    cv.imshow('Face',frame)
    k = cv.waitKey(30) & 0xff
    if k==27:
        break

cv.release()
cv.destroyAllWindows()

