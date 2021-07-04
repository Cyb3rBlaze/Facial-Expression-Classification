import tensorflow as tf
import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
import urllib.request as urlreq
import os

def run_inference(model, sample):
    return model.predict((sample))


def preprocess(frame):
    img = np.array(frame, dtype='uint8')
    img = cv2.resize(img, (48, 48))
    return np.float64((img).reshape((1, 48, 48, 1)))/255

model = tf.keras.models.load_model("model3")

model.summary()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

classes = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

while True:
    ret, frame = vid.read()

    detect_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(detect_face, 1.1, 5)

    output = None

    found_face = False

    for (x, y, w, h) in faces:
        extracted = preprocess(detect_face[y-30:y+h+30, x-30:x+w+30])

        output = run_inference(model, extracted)
        found_face = True

        cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0), 2)

        break


    if found_face == True:
        max_val = max(output[0])
        max_val_index = list(output[0]).index(max_val)

        true_class = classes[max_val_index]

        print(true_class)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()

