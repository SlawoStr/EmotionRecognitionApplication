#pragma once
#include "pyhelper.h"
#include <vector>

class PythonConnector
{
public:
	PythonConnector();
	~PythonConnector();
private:
	CPyInstance hInstance;
	CPyObject pythonClass;
	CPyObject object;
	bool ErrorCondition;
public:
	int detectEmotions(const std::string & imagepath, const int & faceDetector,const int & classificator);
	int getNumberOfFaces();
	void getFace(int);
	void getFeatures(int);
	void resetPredictor();
};