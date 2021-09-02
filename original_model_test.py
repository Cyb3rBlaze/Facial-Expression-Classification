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
    return np.float32((img).reshape((1, 48, 48, 1)))/255

model = tf.keras.models.load_model("model")

model.summary()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

classes = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger"}

while True:
    ret, frame = vid.read()

    detect_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(detect_face, 1.1, 5)

    output = None

    found_face = False

    for (x, y, w, h) in faces:
        try:
            extracted = preprocess(detect_face[y-30:y+h+30, x-30:x+w+30])

            output = run_inference(model, extracted)
            found_face = True

            cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0), 2)
            for i in range(len(output[0])):
                output[0][i] = round(output[0][i], 3)
            output_ = {"neut": output[0][0], "happ": output[0][1], "surp": output[0][2], "sad": output[0][3], "ang": output[0][4]}
            frame = cv2.putText(frame, str(output_), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            max_val = max(output[0])
            max_val_index = list(output[0]).index(max_val)

            true_class = classes[max_val_index]

            average_index = round(average_emotion)

            average_class = classes[average_index]

            frame = cv2.putText(frame, str(true_class), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        except:
            pass        

        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()

