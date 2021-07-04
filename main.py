import tensorflow as tf
import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
import urllib.request as urlreq
import os

Resize_pixelsize = 197
DROPOUT_RATE = 0.5

def create_model():
    vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=(Resize_pixelsize, Resize_pixelsize, 3), pooling='avg')
    last_layer = vgg_notop.get_layer('avg_pool').output
    x = tf.keras.layers.Flatten(name='flatten')(last_layer)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc6')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc7')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    out = tf.keras.layers.Dense(7, activation='softmax', name='classifier')(x)

    return tf.keras.Model(vgg_notop.input, out)

def run_inference(model, sample):
    return model.predict((sample))

LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

def preprocess(img):
    img = np.array(img, dtype='uint8')
    _, landmarks = landmark_detector.fit(img, np.array([[1, 1, 46, 46]]))
    blank_image = np.zeros((48,48,3), np.uint8)
    for landmark in landmarks:
            for x,y in landmark[0]:
                    if y > 47:
                            y = 47
                    if x > 47:
                            x = 47
                    blank_image[int(y),int(x)] = (255, 255, 255)
    landmarked_image = np.float64(cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY).reshape((1, 48, 48, 1)))/255
    x_coord = np.zeros((48, 48))
    y_coord = np.zeros((48, 48))
    for i in range(48):
        y_coord[i] = np.full((48), i+1)
    x_coord = y_coord.T
    final_output = np.concatenate((landmarked_image, np.float64(x_coord.reshape((1, 48, 48, 1)))/48, np.float64(y_coord.reshape((1, 48, 48, 1)))/48, np.float64(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(1, 48, 48, 1))/255), axis=3)
    return final_output

model = tf.keras.models.load_model("model6")

model.summary()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

classes = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

while True:
    ret, frame = vid.read()

    detect_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = cv2.cvtColor(detect_face, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(detect_face, 1.1, 5)

    output = None

    found_face = False

    for (x, y, w, h) in faces:
        extracted = preprocess(cv2.resize(detect_face[y-30:y+h+30, x-30:x+w+30], (48, 48)))

        output = run_inference(model, extracted)
        print(output)
        found_face = True

        cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()

