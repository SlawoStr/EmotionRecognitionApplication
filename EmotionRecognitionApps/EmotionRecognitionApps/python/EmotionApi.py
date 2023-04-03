from FaceDetectors import HOGImageTesting, DNNTesting, HaarCascadeTesting
from imutils import face_utils
import cv2
import dlib
from keras.models import load_model
import numpy as np
import math
import pickle

predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]  # Emotion list


def detectFeaturePoints(gray, face):
    landmarks = predictor(gray, face)
    shape = face_utils.shape_to_np(landmarks)
    return shape


def predictSVMEmotions(coordinatesList, model):
    landmarks_vectorised = []
    for x, y in coordinatesList:
        landmarks_vectorised.append(x)
        landmarks_vectorised.append(y)
    data_list = [landmarks_vectorised]
    numpy_data = np.array(data_list)
    result = model.predict_proba(numpy_data)
    return result


def predictCNNEmotions(greyImage, model):
    img = cv2.resize(greyImage, (48, 48), interpolation=cv2.INTER_AREA)
    img = np.vstack(img).reshape(-1, 48, 48, 1)
    result = model.predict(img)
    return result


class EmotionPredictor:
    faceCoordinates = []
    featuresLocation = []
    predictionsProbabilities = []
    currentImage = []

    def __init__(self):
        self.DNNModel = load_model('resources/models/DNNModel.h5')
        self.CNNModel = load_model('resources/models/CNNModel.h5')
        self.SVMModel = pickle.load(open('resources/models/SVMModel.sav', 'rb'))

    def predictEmotion(self, filepath, facedetector, emotionClassifier):
        image = cv2.imread(filepath)
        self.currentImage = image

        featuresImage = self.currentImage.copy()
        emotionsImage = self.currentImage.copy()

        if facedetector == 1:
            self.faceCoordinates = HaarCascadeTesting(image)
        elif facedetector == 2:
            self.faceCoordinates = HOGImageTesting(image)
        elif facedetector == 3:
            self.faceCoordinates = DNNTesting(image)
        # DNN and SVM
        if len(self.faceCoordinates) == 0:
            return -1
        for i in range(0, len(self.faceCoordinates)):
            face = dlib.rectangle(self.faceCoordinates[i][0], self.faceCoordinates[i][1], self.faceCoordinates[i][2],
                                  self.faceCoordinates[i][3])
            shape = detectFeaturePoints(image, face)

            emotionsImage = cv2.rectangle(emotionsImage, (
                self.faceCoordinates[i][0], self.faceCoordinates[i][1]), (self.faceCoordinates[i][2],
                                                                          self.faceCoordinates[i][3]), (255, 0, 0), 2)

            xlist = []
            ylist = []

            for (x, y) in shape:
                xlist.append(float(x))
                ylist.append(float(y))
                cv2.circle(featuresImage, (x, y), 2, (255, 0, 0), 2)

            self.featuresLocation.append(shape)

            left_eye = [shape[36], shape[39]]
            right_eye = [shape[42], shape[45]]

            left_midpoint = [(left_eye[0][0] + left_eye[1][0]) / 2, (left_eye[0][1] + left_eye[1][1]) / 2]
            right_midpoint = [(right_eye[0][0] + right_eye[1][0]) / 2, (right_eye[0][1] + right_eye[1][1]) / 2]

            distance = int(
                (((left_midpoint[0] - right_midpoint[0]) ** 2) + ((left_midpoint[1] - right_midpoint[1]) ** 2)) ** 0.5)

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x - xmean) for x in xlist]
            ycentral = [(y - ymean) for y in ylist]

            landmarks_vectorised = []

            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                meannp = np.asarray((ymean, xmean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp - meannp)
                dist = dist * 100 / distance
                landmarks_vectorised.append((int(dist), int((math.atan2(y, x) * 360) / (2 * math.pi))))

            # SVM
            if emotionClassifier == 1:
                results = predictSVMEmotions(landmarks_vectorised, self.SVMModel)
                self.predictionsProbabilities.append(results)
            # DNN
            elif emotionClassifier == 2:
                testingValue = np.array(landmarks_vectorised)
                testingValue = np.expand_dims(testingValue, 0)
                results = self.DNNModel.predict(testingValue)
                self.predictionsProbabilities.append(results)
            # CNN
            elif emotionClassifier == 3:
                face2 = image[self.faceCoordinates[i][1]:self.faceCoordinates[i][3],
                        self.faceCoordinates[i][0]:self.faceCoordinates[i][2]]
                face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
                results = predictCNNEmotions(face2, self.CNNModel)
                self.predictionsProbabilities.append(results)
            emotion = np.argmax(results)
            cv2.putText(emotionsImage, emotions[emotion], (self.faceCoordinates[i][0], self.faceCoordinates[i][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite("results/allFeatures.jpg", featuresImage)
        cv2.imwrite("results/allEmotions.jpg", emotionsImage)
        return 1

    def resetPredictor(self):
        self.faceCoordinates.clear()
        self.predictionsProbabilities.clear()
        self.featuresLocation.clear()
        #self.currentImage = None

    def getNumberOfFaces(self):
        return len(self.faceCoordinates)

    def getFeatures(self, faceNumber):
        face = cv2.imread('results/allFeatures.jpg')[
               self.faceCoordinates[faceNumber][1]:self.faceCoordinates[faceNumber][3],
               self.faceCoordinates[faceNumber][0]:self.faceCoordinates[faceNumber][2]].copy()
        cv2.imwrite("results/featureFace.jpg", face)

    def getEmotion(self, faceNumber):
        face = cv2.imread('results/allEmotions.jpg')[
               self.faceCoordinates[faceNumber][1]:self.faceCoordinates[faceNumber][3],
               self.faceCoordinates[faceNumber][0]:self.faceCoordinates[faceNumber][2]].copy()
        cv2.imwrite("results/emotionFace.jpg", face)
