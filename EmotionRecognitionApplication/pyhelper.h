#pragma once
#pragma once

#include <Python.h>

class CPyInstance
{
public:
	CPyInstance()
	{
		//You can check required paths by starting certain python script in pycharm. On the top screen of terminal you will see all paths that were used to start script. Copy it and add to this path. Also you must add path to folder(in this its relase)
		Py_SetPath(L"E:\\Github\\EmotionRecognitionApplication\\EmotionRecognitionApplication\\Python;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\python38.zip;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\DLLs;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\lib;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow;C:\\Users\\SSSS\\AppData\\Roaming\\Python\\Python38\\site-packages;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\lib\\site-packages;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\lib\\site-packages\\win32;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\lib\\site-packages\\win32\\lib;C:\\Users\\SSSS\\miniconda3\\envs\\tensorflow\\lib\\site-packages\\Pythonwin");
		Py_Initialize();
	}
	~CPyInstance()
	{
		Py_Finalize();
	}
};


class CPyObject
{
private:
	PyObject* p;
public:
	CPyObject() : p(NULL)
	{}

	CPyObject(PyObject* _p) : p(_p)
	{}


	~CPyObject()
	{
		Release();
	}

	PyObject* getObject()
	{
		return p;
	}

	PyObject* setObject(PyObject* _p)
	{
		return (p = _p);
	}

	PyObject* AddRef()
	{
		if (p)
		{
			Py_INCREF(p);
		}
		return p;
	}

	void Release()
	{
		if (p)
		{
			Py_DECREF(p);
		}

		p = NULL;
	}

	PyObject* operator ->()
	{
		return p;
	}

	bool is()
	{
		return p ? true : false;
	}

	operator PyObject* ()
	{
		return p;
	}

	PyObject* operator = (PyObject* pp)
	{
		p = pp;
		return p;
	}

	operator bool()
	{
		return p ? true : false;
	}
};

