#pragma once
#include "PythonConnector.h"
#include "WidgetManager.h"

class Application
{
public:
	Application();
	void run();
private:
	void draw();
	void update();
	void pollEvent();
private:
	sf::RenderWindow window;
	PythonConnector pyConnector;
	WidgetManager widManager;
};