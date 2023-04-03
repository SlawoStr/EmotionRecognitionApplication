import cv2
from enum import Enum
import dlib
import numpy as np
import math
from EmotionRecognitionAlgorithms.FaceDetectors import GetDNNFace, GetHogFace, GetHaarCascadeFace
from EmotionRecognitionAlgorithms.FeatureDetector import getDLIBFeaturePoints
from EmotionRecognitionAlgorithms.EmotionPredictor import predictSVMEmotions, predictCNNEmotions, predictDNNEmotions


class Emotions(Enum):
    ANGER = 1
    DISGUST = 2
    FEAR = 3
    HAPPINESS = 4
    SADNESS = 5
    SURPRISE = 6
    NEUTRAL = 7


def getEmotion(predictions):
    emotions = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'SADNESS', 5: 'SURPRISE', 6: 'NEUTRAL'}
    return emotions[int(np.argmax(predictions))]


class EmotionPredictor:
    def __init__(self):
        self.currentImage = None
        # Function
        self.facePredictor = GetHaarCascadeFace
        self.featureDetector = getDLIBFeaturePoints
        self.emotionClassifier = 3
        # Results
        self.faceCoordinates = []
        self.emotionPredictions = []
        self.landmarksCoordinates = []

    def runPredictor(self, filepath):
        self.currentImage = cv2.imread(filepath)
        # File doesnt exist or bad format
        if self.currentImage is None:
            return "Bad file"
        self.faceCoordinates = self.facePredictor(self.currentImage)
        # File doesnt contain any faces (or algorithm couldn't find them)
        if len(self.faceCoordinates) == 0:
            return "No faces on this image"

        for x, y, w, z in self.faceCoordinates:
            face = dlib.rectangle(x, y, w, z)
            shape = self.featureDetector(self.currentImage, face)
            # Save location of all face landmarks
            self.landmarksCoordinates.append(shape)

            xList = []
            yList = []

            for xCord, yCord in shape:
                xList.append(xCord)
                yList.append(yCord)

            # CNN
            if self.emotionClassifier == 1:
                currFace = self.currentImage[y:z, x:w]
                currFace = cv2.cvtColor(currFace, cv2.COLOR_BGR2GRAY)
                self.emotionPredictions.append(predictCNNEmotions(currFace))
            # Other methods that require landmarks (and normalization)
            else:
                left_eye = [shape[36], shape[39]]
                right_eye = [shape[42], shape[45]]
                left_midpoint = [(left_eye[0][0] + left_eye[1][0]) / 2, (left_eye[0][1] + left_eye[1][1]) / 2]
                right_midpoint = [(right_eye[0][0] + right_eye[1][0]) / 2, (right_eye[0][1] + right_eye[1][1]) / 2]

                distance = int(
                    (((left_midpoint[0] - right_midpoint[0]) ** 2) + (
                            (left_midpoint[1] - right_midpoint[1]) ** 2)) ** 0.5)

                xMean = np.mean(xList)
                yMean = np.mean(yList)
                xCentral = [(x - xMean) for x in xList]
                yCentral = [(y - yMean) for y in yList]

                landmarksVectorized = []
                for xC, yC, wL, zL in zip(xCentral, yCentral, xList, yList):
                    meanNp = np.asarray((yMean, xMean))
                    cordsNp = np.asarray((zL, wL))
                    dist = np.linalg.norm(cordsNp - meanNp)
                    dist = dist * 100 / distance

                    landmarksVectorized.append((int(dist), int((math.atan2(y, x) * 360) / (2 * math.pi))))
                # SVM Model
                if self.emotionClassifier == 2:
                    self.emotionPredictions.append(predictSVMEmotions(landmarksVectorized))
                # DNN Model
                elif self.emotionClassifier == 3:
                    self.emotionPredictions.append(predictDNNEmotions(landmarksVectorized))

    def setFaceDetector(self, faceDetectorID):
        if faceDetectorID == 1:
            self.facePredictor = GetHaarCascadeFace
        elif faceDetectorID == 2:
            self.facePredictor = GetDNNFace
        elif faceDetectorID == 3:
            self.facePredictor = GetHogFace
        # Default Haar Cascade
        else:
            self.facePredictor = GetHaarCascadeFace

    def setLandmarkDetector(self, landmarkDetectorID):
        if landmarkDetectorID == 1:
            self.featureDetector = getDLIBFeaturePoints
        # Currently only one features detector
        else:
            self.featureDetector = getDLIBFeaturePoints

    def setEmotionClassifier(self, classifierID):
        if classifierID == 1:
            self.emotionClassifier = 1
        elif classifierID == 2:
            self.emotionClassifier = 2
        elif classifierID == 3:
            self.emotionClassifier = 3
        else:
            self.emotionClassifier = 1

    def resetPredictor(self):
        self.faceCoordinates.clear()
        self.emotionPredictions.clear()
        self.landmarksCoordinates.clear()

    def printImage(self, displayMode, faceNumber=1):
        # Didn't detect any faces on this image
        if len(self.faceCoordinates) == 0:
            cv2.imshow('image', self.currentImage)
            cv2.waitKey(0)
            return

        # Full image with face detection and emotions
        if displayMode == 1:
            image = self.currentImage.copy()
            counter = 0
            for x, y, w, z in self.faceCoordinates:
                cv2.rectangle(image, (x, y), (w, z), (255, 0, 0), 2)
                result = getEmotion(self.emotionPredictions[counter])
                cv2.putText(image, result, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('image', image)
                cv2.waitKey(0)
                counter += 1
        # Full image with landmarks
        elif displayMode == 2:
            image = self.currentImage.copy()
            for i in range(len(self.landmarksCoordinates)):
                for x, y in self.landmarksCoordinates[i]:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            cv2.imshow('image', image)
            cv2.waitKey(0)
        # Emotion detection for face with faceNumber = x
        elif displayMode == 3:
            if faceNumber >= len(self.faceCoordinates):
                faceNumber = 0
            face = self.currentImage[self.faceCoordinates[faceNumber][1]:self.faceCoordinates[faceNumber][3],
                   self.faceCoordinates[faceNumber][0]:self.faceCoordinates[faceNumber][2]]
            result = getEmotion(self.emotionPredictions[faceNumber])
            cv2.putText(face, result, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('image', face)
            cv2.waitKey(0)
        # Landmarks detection for face with faceNumber = x
        elif displayMode == 4:
            if faceNumber >= len(self.faceCoordinates):
                faceNumber = 0
            image = self.currentImage.copy()
            for i in range(len(self.landmarksCoordinates)):
                for x, y in self.landmarksCoordinates[i]:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            face = image[self.faceCoordinates[faceNumber][1]:self.faceCoordinates[faceNumber][3],
                   self.faceCoordinates[faceNumber][0]:self.faceCoordinates[faceNumber][2]]
            cv2.imshow('image', face)
            cv2.waitKey(0)


predictor = EmotionPredictor()
predictor.runPredictor("facehp.jpg")
predictor.printImage(4)
