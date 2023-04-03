import cv2
import dlib
import numpy as np


def GetHaarCascadeFace(image):
    faceCascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')
    grayFace = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFace, 1.3, 5)
    # Transform to face coordinates
    facesCoordinates = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
    return facesCoordinates


def GetDNNFace(image):
    configFile = 'resources/deploy.prototxt.txt'
    modelFile = 'resources/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    facesCoordinates = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            facesCoordinates.append((startX, startY, endX, endY))
    return facesCoordinates


def GetHogFace(image):
    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRectangles = face_detector(gray, 1)
    faceCoordinates = []
    for face in faceRectangles:
        faceCoordinates.append((face.left(), face.top(), face.right(), face.bottom()))
    return faceCoordinates
