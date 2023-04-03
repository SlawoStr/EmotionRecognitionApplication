import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle

emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]  # Emotion list
clf = SVC(kernel='linear', probability=True,
          tol=1e-3)
globvar = 0


def loadData(filepath):
    f = open(filepath, "r")
    data = f.read()
    emotion = np.array(data.split())
    f.close()
    values = []
    emotion_list = []
    for i in range(0, len(emotion)):
        values.append(emotion[i])
        if len(values) == 72:
            emotion_list.append(values.copy())
            values.clear()
    return emotion_list


def predictEmotion(coordinatesList, plot=False):
    data_list = [coordinatesList]
    numpy_data = np.array(data_list)
    result = clf.predict_proba(numpy_data)
    if plot:
        left = [1, 2, 3, 4, 5, 6]

        # heights of bars
        height = []
        for i in range(0, 6):
            height.append(result[0][i] * 100)

        # plotting a bar chart
        plt.bar(left, height, tick_label=emotions,
                width=0.5, color=['blue', 'yellow', 'green', 'red', 'purple', 'red'])

        # naming the x-axis
        plt.xlabel('Emotion')
        # naming the y-axis
        plt.ylabel('Probability')
        # plot title
        plt.title('Emotion Classification')

        # function to show the plot
        plt.show()
    return emotions[np.argmax(result)]


def trainSVM(load=True):
    if load:
        loadModel()
    else:
        anger_list = loadData("resources/trainingData/anger.txt")
        disgust_list = loadData("resources/trainingData/disgust.txt")
        fear_list = loadData("resources/trainingData/fear.txt")
        happiness_list = loadData("resources/trainingData/happiness.txt")
        sadness_list = loadData("resources/trainingData/sadness.txt")
        surprise_list = loadData("resources/trainingData/surprise.txt")

        result_matrix = [[0, 0, 0, 0, 0, 0, ], [0, 0, 0, 0, 0, 0, ], [0, 0, 0, 0, 0, 0, ], [0, 0, 0, 0, 0, 0, ],
                         [0, 0, 0, 0, 0, 0, ], [0, 0, 0, 0, 0, 0, ]]

        training_data = []
        training_lables = []
        prediction_data = []
        prediciton_lables = []

        random_list = random.sample(range(0, len(anger_list)), int(len(anger_list) * 0.2))

        for i in range(0, len(anger_list)):
            if i in random_list:
                prediction_data.append(anger_list[i])
                prediciton_lables.append(0)
            else:
                training_data.append(anger_list[i])
                training_lables.append(0)

        random_list = random.sample(range(0, len(disgust_list)), int(len(disgust_list) * 0.2))

        for i in range(0, len(disgust_list)):
            if i in random_list:
                prediction_data.append(disgust_list[i])
                prediciton_lables.append(1)
            else:
                training_data.append(disgust_list[i])
                training_lables.append(1)

        random_list = random.sample(range(0, len(fear_list)), int(len(fear_list) * 0.2))

        for i in range(0, len(fear_list)):
            if i in random_list:
                prediction_data.append(fear_list[i])
                prediciton_lables.append(2)
            else:
                training_data.append(fear_list[i])
                training_lables.append(2)

        random_list = random.sample(range(0, len(happiness_list)), int(len(happiness_list) * 0.2))

        for i in range(0, len(happiness_list)):
            if i in random_list:
                prediction_data.append(happiness_list[i])
                prediciton_lables.append(3)
            else:
                training_data.append(happiness_list[i])
                training_lables.append(3)

        random_list = random.sample(range(0, len(sadness_list)), int(len(sadness_list) * 0.2))

        for i in range(0, len(sadness_list)):
            if i in random_list:
                prediction_data.append(sadness_list[i])
                prediciton_lables.append(4)
            else:
                training_data.append(sadness_list[i])
                training_lables.append(4)

        random_list = random.sample(range(0, len(surprise_list)), int(len(surprise_list) * 0.2))

        for i in range(0, len(surprise_list)):
            if i in random_list:
                prediction_data.append(surprise_list[i])
                prediciton_lables.append(5)
            else:
                training_data.append(surprise_list[i])
                training_lables.append(5)

        npar_train = np.array(training_data)
        npar_trainlabs = np.array(training_lables)

        print("training SVM linear")  # train SVM
        clf.fit(npar_train, npar_trainlabs)

        npar_pred = np.array(prediction_data)
        pred_lin = clf.score(npar_pred, prediciton_lables)
        print("linear: ", pred_lin)


        title_options = [("Confusion matrix", 'true')]
        for title, normalize in title_options:
            disp = plot_confusion_matrix(clf, npar_pred, prediciton_lables, display_labels=emotions, cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
        plt.show()
        saveModel()



def saveModel():
    filename = 'SVMModel.sav'
    pickle.dump(clf, open(filename, 'wb'))


def loadModel():
    filename = 'SVMModel.sav'
    global clf
    clf = pickle.load(open(filename, 'rb'))
