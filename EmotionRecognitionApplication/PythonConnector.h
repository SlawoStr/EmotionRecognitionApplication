#pragma once
#include "pyhelper.h"
#include <vector>
#include <string>

class PythonConnector
{
public:
	PythonConnector();
	int testFunction();
private:
	CPyInstance hInstance;
	CPyObject pythonClass;
	CPyObject object;
	bool ErrorCondition;
};