from keras.models import load_model
import pickle
import numpy as np
import cv2

CNNModel = load_model('resources/models/CNNModel.h5')
DNNModel = load_model('resources/models/DNNModel.h5')
SVMModel = pickle.load(open('resources/models/SVMModel.sav', 'rb'))


def predictSVMEmotions(landmarks):
    landmarksVectorized = []
    for x, y in landmarks:
        landmarksVectorized.append(x)
        landmarksVectorized.append(y)
    numpy_data = np.array([landmarksVectorized])
    return SVMModel.predict_proba(numpy_data)


def predictCNNEmotions(grayImg):
    img = cv2.resize(grayImg, (48, 48), interpolation=cv2.INTER_AREA)
    img = np.vstack(img).reshape(-1, 48, 48, 1)
    return CNNModel.predict(img)


def predictDNNEmotions(landmarksVectorized):
    landmarks = np.array(landmarksVectorized)
    landmarks = np.expand_dims(landmarks, 0)
    return DNNModel.predict(landmarks)


