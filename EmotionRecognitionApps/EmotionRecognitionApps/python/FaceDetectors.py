import cv2
import dlib
import numpy as np


def HaarCascadeTesting(image):
    faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    faces_coordinates = []
    for (x, y, w, h) in faces:
        faceX = x
        faceY = y
        faceW = x + w
        faceH = y + h
        faces_coordinates.append((faceX, faceY, faceW, faceH))
    return faces_coordinates


def DNNTesting(image):
    configFile = 'resources/deploy.prototxt.txt'
    modelFile = 'resources/res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces_coordinates = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faceX = startX
            faceY = startY
            faceW = endX
            faceH = endY
            faces_coordinates.append((faceX, faceY, faceW, faceH))
    return faces_coordinates


def HOGImageTesting(image):
    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    face_coordinates = []
    for (i, rect) in enumerate(rects):
        faceX = rect.left()
        faceY = rect.top()
        faceH = rect.right()
        faceW = rect.bottom()
        face_coordinates.append((faceX, faceY, faceH, faceW))
    return face_coordinates
