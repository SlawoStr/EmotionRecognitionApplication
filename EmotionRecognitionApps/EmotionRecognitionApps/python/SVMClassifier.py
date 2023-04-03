import cv2
import glob
import random
import math
import numpy as np
import dlib
from sklearn.svm import SVC
import pickle

emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]  # Emotion list
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)
data = {}


def calculatePointDistance(point1, point2):
    result = abs(point1[1] - point2[1])
    return result


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("resources\\images\\Oryginal\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def get_landmarks2(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def get_landmarks(image):
    height = np.size(image, 0)
    width = np.size(image, 1)
    face_rects = [dlib.rectangle(left=0, top=0, right=height - 1, bottom=width - 1)]
    shape = get_landmarks2(image, face_rects)
    xlist = []
    ylist = []

    for (x, y) in shape:
        xlist.append(float(x))
        ylist.append(float(y))

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
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
    data['landmarks_vectorised'] = landmarks_vectorised


def saveModel():
    filename = 'resources/SVMModel.sav'
    pickle.dump(clf, open(filename, 'wb'))


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            get_landmarks(gray)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            get_landmarks(gray)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def trainModel():
    accur_lin = []
    for i in range(0, 1):
        print("Making sets %s" % i)  # Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = make_sets()

        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        print("training SVM linear %s" % i)  # train SVM
        clf.fit(npar_train, training_labels)

        print("getting accuracies %s" % i)  # Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        pred_lin = clf.score(npar_pred, prediction_labels)
        print("linear: ", pred_lin)
        accur_lin.append(pred_lin)  # Store accuracy in a list
        saveModel()
    print("Mean value lin svm: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs
trainModel()