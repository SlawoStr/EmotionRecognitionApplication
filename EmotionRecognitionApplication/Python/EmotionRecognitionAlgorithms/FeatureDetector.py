from imutils import face_utils
import dlib


def getDLIBFeaturePoints(image, face):
    predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(image, face)
    shape = face_utils.shape_to_np(landmarks)
    return shape
