import cv2
import os
import numpy as np
import math
import imutils
import time
from SVMClassifier import predictEmotion, trainSVM
from matplotlib import pyplot as plt


def haarCascadeImageTesting(filepath):
    faceCascade = cv2.CascadeClassifier("resources/frontalface_detector.xml")
    img = cv2.imread(filepath)
    # img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        faces_coordinates = []
        for (x, y, w, h) in faces:
            faceX = x
            faceY = y
            faceW = x + w
            faceH = y + h
            faces_coordinates.append((faceX, faceY, faceW, faceH))
        return img, faces_coordinates
    return None


def rotateImage(face):
    eyes = getEyes(face, True)
    if len(eyes[0]) < 2 or len(eyes[1]) < 2:
        return None
    angle = math.atan(math.fabs(eyes[0][1] - eyes[1][1]) / (eyes[0][0] - eyes[1][0]))
    degree = math.degrees(angle)
    if eyes[0][1] < eyes[1][1]:
        degree = degree * -1
    rotated_face = imutils.rotate(face, degree)
    return rotated_face


def cutEyebrows(eye):
    height, width = eye.shape[:2]
    eyebrow_h = int(height / 4)
    eye = eye[eyebrow_h:height, 0:width]
    return eye, eyebrow_h


def getSVMInput(faceCoordinates, eyes):
    xList = []
    yList = []
    currentNumber = 0
    for i in range(0, 18):
        xList.append(faceCoordinates[currentNumber][0])
        yList.append(faceCoordinates[currentNumber][1])
        currentNumber = currentNumber + 1
    eyeDistance = int((((eyes[1][0] - eyes[0][0]) ** 2) + ((eyes[1][1] - eyes[0][1]) ** 2)) ** 0.5)
    xmean = np.mean(xList)
    ymean = np.mean(yList)

    xCentral = [(x - xmean) for x in xList]
    yCentral = [(y - ymean) for y in yList]

    landmarks_vectorised = []

    for x, y, w, z in zip(xCentral, yCentral, xList, yList):
        meannp = np.asarray((ymean, xmean))
        coornp = np.asarray((z, w))
        dist = np.linalg.norm(coornp - meannp)
        dist = dist * 100 / eyeDistance
        w = w * 100 / eyeDistance
        z = z * 100 / eyeDistance
        landmarks_vectorised.append(w)
        landmarks_vectorised.append(z)
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))
    return landmarks_vectorised


def calculateThreshold(image_to_process):
    initial_threshold = 50
    height = np.size(image_to_process, 0)
    width = np.size(image_to_process, 1)
    while 1:
        intensity_below = []
        intensity_above = []
        for i in range(0, height):
            for j in range(0, width):
                if image_to_process[i, j] < initial_threshold:
                    intensity_below.append(image_to_process[i, j])
                else:
                    intensity_above.append(image_to_process[i, j])
        new_threshold = (np.mean(intensity_below) + np.mean(intensity_above)) / 2
        if int(new_threshold) == initial_threshold:
            break
        else:
            initial_threshold = int(new_threshold)
    return initial_threshold


def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def contrastStretching(image):
    pixelVal_vec = np.vectorize(pixelVal)

    r1 = np.min(image) + (np.mean(image) - np.min(image)) / 3 * 2
    s1 = 0
    r2 = np.mean(image) + (np.max(image) - np.mean(image)) / 3
    s2 = 255
    contrast_stretched = pixelVal_vec(image, r1, s1, r2, s2)
    cv2.imwrite("example.jpg", contrast_stretched)
    image = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
    return image


def contrastStretching2(image):
    pixelVal_vec = np.vectorize(pixelVal)

    r1 = np.min(image) + (np.mean(image) - np.min(image)) / 3 + 5
    s1 = 0
    r2 = np.mean(image)
    s2 = 255
    contrast_stretched = pixelVal_vec(image, r1, s1, r2, s2)
    cv2.imwrite("example.jpg", contrast_stretched)
    image = cv2.imread("example.jpg", cv2.IMREAD_GRAYSCALE)
    return image


def getEyes(face, rotate=False):
    eye_cascade = cv2.CascadeClassifier("resources/eye_detector.xml")
    detected_eyes = eye_cascade.detectMultiScale(face)
    height = np.size(face, 0)
    width = np.size(face, 1)
    points_r = []
    points_l = []
    leftEye = None
    rightEye = None
    eye_coordinatesL = []
    eye_coordinatesR = []
    for (ex, ey, ew, eh) in detected_eyes:
        eyeCenter = ex + ew / 2
        if ey > height / 2:
            pass
        elif eyeCenter < width * 0.5:
            leftEye = face[ey:ey + ew, ex:ex + eh]
            points_l.append(int(ex + ew / 2))
            points_l.append(int(ey + eh / 2))
            eye_coordinatesL = (ex, ey)
        else:
            rightEye = face[ey:ey + ew, ex:ex + eh]
            points_r.append(int(ex + ew / 2))
            points_r.append(int(ey + eh / 2))
            eye_coordinatesR = (ex, ey)
    if rotate:
        return points_l, points_r
    else:
        eyes = [leftEye, rightEye]
        eyes_coordinates = [eye_coordinatesL, eye_coordinatesR]
        mid_points = [(points_l[0], points_l[1]), (points_r[0], points_r[1])]
        return eyes, eyes_coordinates, mid_points


def detectEyebrows(image, eyes):
    result = int((((eyes[1][0] - eyes[0][0]) ** 2) + ((eyes[1][1] - eyes[0][1]) ** 2)) ** 0.5)
    result2 = result * 0.33
    eyebrows = []
    eyebrowCoordinates = []
    rec_point = (int(eyes[0][0]), int(eyes[0][1] - result2))
    left_eye_points = [int(rec_point[0] - (result / 8 * 3)), int(rec_point[1] - result2 / 4),
                       int(rec_point[0] + (result / 8 * 3)), int(rec_point[1] + result2 / 2)]

    rec_point = (int(eyes[1][0]), int(eyes[1][1] - result2))
    right_eye_points = [int(rec_point[0] - (result / 8 * 3)), int(rec_point[1] - result2 / 4),
                        int(rec_point[0] + (result / 8 * 3)), int(rec_point[1] + result2 / 2)]

    eyebrows.append(image[left_eye_points[1]:left_eye_points[3], left_eye_points[0]:left_eye_points[2]])
    eyebrows.append(image[right_eye_points[1]:right_eye_points[3], right_eye_points[0]:right_eye_points[2]])

    eyebrowCoordinates.append((left_eye_points[0], left_eye_points[1]))
    eyebrowCoordinates.append((right_eye_points[0], right_eye_points[1]))
    return eyebrows, eyebrowCoordinates


def detectNose(image, eyes):
    result = int((((eyes[1][0] - eyes[0][0]) ** 2) + ((eyes[1][1] - eyes[0][1]) ** 2)) ** 0.5)
    result2 = result * 0.6

    noseCoordinates = []

    rec_point = (int(eyes[0][0] + result / 2), int(eyes[0][1] + result2))
    nose_position = [int(rec_point[0] - result / 3), int(rec_point[1] - result / 4), int(rec_point[0] + result / 3),
                     int(rec_point[1] + result / 4)]
    nose = image[nose_position[1]:nose_position[3], nose_position[0]:nose_position[2]]
    noseCoordinates.append((nose_position[0], nose_position[1]))
    return nose, noseCoordinates


def detectMouth(image, eyes):
    result = int((((eyes[1][0] - eyes[0][0]) ** 2) + ((eyes[1][1] - eyes[0][1]) ** 2)) ** 0.5)

    result2 = result * 1.1

    rec_point = (int(eyes[0][0] + result / 2), int(eyes[0][1] + result2))

    mouth_position = [int(rec_point[0] - result / 2), int(rec_point[1] - result2 / 4), int(rec_point[0] + result / 2),
                      int(rec_point[1] + result2 / 3)]
    mouth = image[mouth_position[1]:mouth_position[3], mouth_position[0]:mouth_position[2]]
    return mouth, (mouth_position[0], mouth_position[1])


def detectMouthFeatures(mouth):
    grey_mouth = contrastStretching(mouth)

    cv2.bitwise_not(grey_mouth, grey_mouth)

    value = calculateThreshold(grey_mouth)

    _, threshold = cv2.threshold(grey_mouth, value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    minValueX = 10000
    minValueY = 0
    maxValueX = 0
    maxValueY = 0
    contour = []
    mouthCoordinates = []
    for cnt in contours:
        contour = cnt
        for value in cnt:
            if value[0][0] > maxValueX:
                maxValueX = value[0][0]
                maxValueY = value[0][1]
            elif value[0][0] == maxValueX:
                if value[0][1] > maxValueY:
                    maxValueY = value[0][1]
            if value[0][0] < minValueX:
                minValueX = value[0][0]
                minValueY = value[0][1]
            elif value[0][0] == minValueX:
                if value[0][1] > minValueY:
                    minValueY = value[0][1]
        break

    mouthCoordinates.append((minValueX, minValueY))
    mouthCoordinates.append((maxValueX, maxValueY))

    minValueY = 10000
    maxValueY = 0
    middlePoint = minValueX + int((maxValueX - minValueX) / 2)
    for value in contour:
        if value[0][0] == middlePoint:
            if value[0][1] > maxValueY:
                maxValueY = value[0][1]
            if value[0][1] < minValueY:
                minValueY = value[0][1]

    mouthCoordinates.append((middlePoint, minValueY))
    mouthCoordinates.append((middlePoint, maxValueY))

    return mouthCoordinates, threshold


def detectNoseFeatures(grey_nose):
    nose = grey_nose.copy()
    h = nose.shape[0]
    w = nose.shape[1]
    smallest = np.amin(nose)
    for i in range(0, h):
        for j in range(0, w):
            nose[i, j] = nose[i, j] - smallest

    biggest = np.amax(nose)
    intensity = 255 / biggest
    for i in range(0, h):
        for j in range(0, w):
            nose[i, j] = nose[i, j] * intensity

    average_intensity = np.mean(nose) / 3
    _, threshold = cv2.threshold(nose, average_intensity, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, None, iterations=1)
    noseCoordinates = []
    img_col_sums = np.sum(threshold, axis=0)
    maxValue = img_col_sums[0]
    worst_column = 0
    worst_column2 = 0
    i = 0
    j = 0
    secondPeak = False
    for value in img_col_sums:
        if value < maxValue:
            j = 0
            maxValue = value
            if secondPeak:
                worst_column2 = i
            else:
                worst_column = i
        elif value == maxValue:
            j = j + 1
        elif value - maxValue > 700 and secondPeak is False:
            secondPeak = True
            maxValue = img_col_sums[0]
            worst_column = worst_column + int(j / 2)
        i = i + 1
    j = 0
    for value in img_col_sums:
        if value == img_col_sums[worst_column2]:
            j = j + 1
    worst_column2 = worst_column2 + int(j / 2)

    leftNosePointX = worst_column
    leftNosePointY = 0
    rightNosePointX = worst_column2
    rightNosePointY = 0
    for i in range(0, np.size(threshold, 0)):
        if threshold[i, worst_column] == 0:
            leftNosePointY = i
    for i in range(0, np.size(threshold, 0)):
        if threshold[i, worst_column2] == 0:
            rightNosePointY = i

    noseCoordinates.append((leftNosePointX, leftNosePointY))
    noseCoordinates.append((rightNosePointX, rightNosePointY))

    return noseCoordinates, threshold


def detectEyeFeatures(eye):
    grey_eye = contrastStretching2(eye)

    value = calculateThreshold(grey_eye)
    _, threshold = cv2.threshold(grey_eye, value, 255, cv2.THRESH_BINARY)

    cv2.bitwise_not(threshold, threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    eyeCoordinates = []
    minValueX = 10000
    minValueY = 0
    maxValueX = 0
    maxValueY = 0
    contour = []
    for cnt in contours:
        contour = cnt
        for value in cnt:
            if value[0][0] > maxValueX:
                maxValueX = value[0][0]
                maxValueY = value[0][1]
            elif value[0][0] == maxValueX:
                if value[0][1] > maxValueY:
                    maxValueY = value[0][1]
            if value[0][0] < minValueX:
                minValueX = value[0][0]
                minValueY = value[0][1]
            elif value[0][0] == minValueX:
                if value[0][1] > minValueY:
                    minValueY = value[0][1]
        break

    eyeCoordinates.append((minValueX, minValueY))
    eyeCoordinates.append((maxValueX, maxValueY))

    section = int((maxValueX - minValueX) / 3)
    sectionLeft = minValueX + section
    sectionRight = minValueX + section * 2
    maxDistance = 0
    maxDistanceIndex = 0

    highestPointY = 0
    lowestPointsY = 10000

    while sectionLeft < sectionRight:
        for value in contour:
            if value[0][0] == sectionLeft:
                if value[0][1] > highestPointY:
                    highestPointY = value[0][1]
                if value[0][1] < lowestPointsY:
                    lowestPointsY = value[0][1]
        if highestPointY - lowestPointsY > maxDistance:
            maxDistance = highestPointY - lowestPointsY
            maxDistanceIndex = sectionLeft
        sectionLeft = sectionLeft + 1

    for value in contour:
        if value[0][0] == sectionLeft:
            if value[0][1] > highestPointY:
                highestPointY = value[0][1]
            if value[0][1] < lowestPointsY:
                lowestPointsY = value[0][1]

    eyeCoordinates.append((maxDistanceIndex, lowestPointsY))
    eyeCoordinates.append((maxDistanceIndex, highestPointY))

    return eyeCoordinates, threshold


def detectEyebrowFeatures(eyebrow):
    grey_eyebrow = eyebrow.copy()
    cv2.bitwise_not(grey_eyebrow, grey_eyebrow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    eyebrow_opened = cv2.morphologyEx(grey_eyebrow, cv2.MORPH_OPEN, kernel)

    newImage = grey_eyebrow - eyebrow_opened

    newImage = contrastStretching(newImage)

    _, th2 = cv2.threshold(newImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    eyebrowCoordinates = []
    minValueX = 10000
    minValueY = 0
    maxValueX = 0
    maxValueY = 0
    for cnt in contours:
        for value in cnt:
            if value[0][0] > maxValueX:
                maxValueX = value[0][0]
                maxValueY = value[0][1]
            elif value[0][0] == maxValueX:
                if value[0][1] > maxValueY:
                    maxValueY = value[0][1]
            if value[0][0] < minValueX:
                minValueX = value[0][0]
                minValueY = value[0][1]
            elif value[0][0] == minValueX:
                if value[0][1] > minValueY:
                    minValueY = value[0][1]
        break

    eyebrowCoordinates.append((minValueX, minValueY))
    eyebrowCoordinates.append((maxValueX, maxValueY))

    return eyebrowCoordinates, th2


def detectEmotions(filepath, landmarkVis=False, emotionVis=False, plotVis=False):
    trainSVM(False)
    for filename in os.listdir(filepath):
        image, faceCoordinates = haarCascadeImageTesting(os.path.join(filepath, filename))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is None or len(faceCoordinates) == 0:
            continue
        for i in range(0, len(faceCoordinates)):

            # Face crop
            face = grayImage[faceCoordinates[i][1]:faceCoordinates[i][3], faceCoordinates[i][0]:faceCoordinates[i][2]]

            # Face rotation
            face = rotateImage(face)
            if face is None:
                continue

            # Face parts
            eyes, eyeCoordinates, eyeMiddlePoints = getEyes(face)
            eyes_region = [cutEyebrows(eyes[0]), cutEyebrows(eyes[1])]
            eyebrows, eyebrowCoordinates = detectEyebrows(face, eyeMiddlePoints)
            nose, noseCoordinates = detectNose(face, eyeMiddlePoints)
            mouth, mouthCoordinates = detectMouth(face, eyeMiddlePoints)
            coordinates_list = []

            # EYEBROWS

            coordinates, eyebrows[0] = detectEyebrowFeatures(eyebrows[0])

            for x, y in coordinates:
                coordinates_list.append((x + eyebrowCoordinates[0][0], y + eyebrowCoordinates[0][1]))

            coordinates, eyebrows[1] = detectEyebrowFeatures(eyebrows[1])

            for x, y in coordinates:
                coordinates_list.append((x + eyebrowCoordinates[1][0], y + eyebrowCoordinates[1][1]))

            # EYES

            coordinates, eyes[0] = detectEyeFeatures(eyes_region[0][0])

            for x, y in coordinates:
                coordinates_list.append((x + eyeCoordinates[0][0], y + eyeCoordinates[0][1] + eyes_region[0][1]))

            coordinates, eyes[1] = detectEyeFeatures(eyes_region[1][0])

            for x, y in coordinates:
                coordinates_list.append((x + eyeCoordinates[1][0], y + eyeCoordinates[1][1] + eyes_region[1][1]))

            # NOSE
            coordinates, nose = detectNoseFeatures(nose)
            for x, y in coordinates:
                coordinates_list.append((x + noseCoordinates[0][0], y + noseCoordinates[0][1]))

            # MOUTH

            coordinates, mouth = detectMouthFeatures(mouth)
            for x, y in coordinates:
                coordinates_list.append((x + mouthCoordinates[0], y + mouthCoordinates[1]))

            for x, y in coordinates_list:
                cv2.circle(face, (x, y), 3, (0, 0, 255), -1, 8)
            emotion = predictEmotion(getSVMInput(coordinates_list, eyeMiddlePoints), plotVis)
            image = cv2.rectangle(image, (faceCoordinates[i][0], faceCoordinates[i][1]),
                                  (faceCoordinates[i][2], faceCoordinates[i][3]),
                                  (255, 0, 0), 2)
            cv2.putText(image, emotion, (faceCoordinates[i][0], faceCoordinates[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
            if landmarkVis:
                cv2.imshow("Face", face)
                cv2.waitKey(0)
        if emotionVis:
            cv2.imshow("image", image)
            cv2.waitKey(0)


folder = "E:/Do zachowania/RemasteredProjects/EmotionRecognition2/EmotionRecognition-Manual/resources/high"  # path to folder with frontal face images
detectEmotions(folder, True, True, True)
#folder2 = "C:/Users/SSSS/Desktop/Nowy folder/EmotionRecognition/resources/images2"
#detectEmotions(folder2, True, False, False)
