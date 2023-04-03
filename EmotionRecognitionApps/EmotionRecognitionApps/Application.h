#pragma once
#include <SFML/Graphics.hpp>
#include "AppUtilities.h"
#include "PythonConnector.h"

class Application
{
public:
	Application();
	~Application();
	void run();
private:
	sf::RenderWindow window;
	sf::Font font;
	PythonConnector connector;
	std::vector<ClassicButton> classicButtonList;
	std::vector<Button> buttonList;
	std::vector<TextArea> textAreaList;
	std::vector<ImageFrame> imageFrameList;
	std::vector<CheckBox> checkBoxList;
	std::vector<InputArea> inputAreaList;
	TextArea errorBox;
	int numberOfFaces;
	int currentFace;
	int displayMode;
	std::string errorMessage;
private:
	void draw();
	void update();
	void pollEvent();
	void handleMouseClick(const sf::Vector2f & mousePos);
};