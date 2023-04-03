# Applications

# 1.Emotion Recognition Application

![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/GUI.PNG)

Application have 3 different face detectors.
1. Haar Cascade Classifier
2. Dlib Hog Face Detector
3. Open cv Deep neutral netowrk face detector

Application have 3 differnet classifiers.
1. SVM (Support vectore machine) - classification depend on localisastion of facial feature points.
2. DNN (Deep neutral network) - same as with svm depend on localisastion of facial feature points
3. CNN (convolutional neutral network) - after face detection area is cut and resized to 48x48.

Facial Feature Points are detected by dlib 68 landmark detector

Application allows to detect emotions on all faces in image
![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/GUI-MultipleFaces.PNG)

For each image application allows to :
1. Check all emotions on image
2. Check all features points on all faces  
![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/Option2.PNG)
3. Check emotion on single face  
![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/Option3.PNG)
4. Check features localiation on single face  
![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/Option4.PNG)

3 Errors are handled:  
1. Empty Path  
2. Wrong Path/extension/file  
3. No faces on image  
![Image of APP](https://github.com/SlawekStr/Applications/blob/main/EmotionRecognitionApps/visualisation/Error.PNG)


# 2.Emotion Recognition- Manual

Project similar to first one but in this facial feature detector is implemented. Face detection - Haar Cascade, classifier - SVM  
NO GUI - only openCV visualisation

