#include "PythonConnector.h"
#include <iostream>

PythonConnector::PythonConnector()
{
	// Python script name without .py
	CPyObject pName = PyUnicode_FromString("EmotionApi");
	// Import module
	CPyObject pModule = PyImport_Import(pName);

	CPyObject dict;

	ErrorCondition = false;
	if (pModule)
	{
		//Get dictionary object that implements module's namespace (__dict__)
		dict = PyModule_GetDict(pModule);
		if (dict == nullptr) {
			PyErr_Print();
			std::cerr << "Fails to get the dictionary.\n";
			ErrorCondition = true;
		}
		else
		{
			// Loading python class
			pythonClass = PyDict_GetItemString(dict, "EmotionPredictor");
			if (pythonClass == nullptr) {
				PyErr_Print();
				std::cerr << "Fails to get the Python class.\n";
				ErrorCondition = true;
			}
			else
			{
				//Check if it is callable
				if (PyCallable_Check(pythonClass)) {
					object = PyObject_CallObject(pythonClass, nullptr);
				}
				else {
					std::cerr << "Cannot instantiate the Python class" << std::endl;
					ErrorCondition = true;
				}
			}
		}
	}
	else
	{
		std::cerr << "ERROR: Module not imported\n" << std::endl;
		ErrorCondition = true;
	}
}

PythonConnector::~PythonConnector()
{
}

int PythonConnector::detectEmotions(const std::string & imagepath, const int & faceDetector, const int & classificator)
{
	CPyObject value = PyObject_CallMethod(object, "predictEmotion", "(s,i,i)", imagepath,faceDetector,classificator);
	return PyLong_AsLong(value);
}

int PythonConnector::getNumberOfFaces()
{
	CPyObject value = PyObject_CallMethod(object, "getNumberOfFaces",NULL);
	return PyLong_AsLong(value);
}

void PythonConnector::getFace(int currentFace)
{
	PyObject_CallMethod(object, "getEmotion", "(i)",currentFace);
}

void PythonConnector::getFeatures(int currentFace)
{
	PyObject_CallMethod(object, "getFeatures", "(i)", currentFace);
}

void PythonConnector::resetPredictor()
{
	PyObject_CallMethod(object, "resetPredictor", NULL);
}
