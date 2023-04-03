#include "Application.h"
#include <iostream>

Application::Application() : window(sf::VideoMode(800.0f, 600.0f), "Emotion Recognition", sf::Style::Close | sf::Style::Titlebar), errorBox(190,30,348,22,"")
{
	if (!font.loadFromFile("Tusz.ttf"))
	{
		std::cout << "ERROR" << std::endl;
	}

	classicButtonList.push_back(ClassicButton(330, 290, 70, 25, "Detect", sf::Color::White));

	buttonList.push_back(Button(730,170,"resources/icons/up.png"));
	buttonList.push_back(Button(730, 186, "resources/icons/down.png"));
	buttonList.push_back(Button(560, 330, "resources/icons/left.png"));
	buttonList.push_back(Button(576, 330, "resources/icons/right.png"));

	textAreaList.push_back(TextArea(130, 30, 38, 25, "Input"));
	textAreaList.push_back(TextArea(560, 30, 48, 25, "Output"));

	textAreaList.push_back(TextArea(35, 360, 38, 25, "Path"));

	textAreaList.push_back(TextArea(30, 420, 118, 25, "Face detection"));
	textAreaList.push_back(TextArea(230, 420, 128, 25, "Features Detection"));
	textAreaList.push_back(TextArea(470, 420, 118, 25, "Classification"));
	textAreaList.push_back(TextArea(650, 420, 38, 25, "Type"));

	textAreaList.push_back(TextArea(70, 470, 118, 18, "Haars Cascade"));
	textAreaList.push_back(TextArea(70, 500, 48, 18, "DNN"));
	textAreaList.push_back(TextArea(70, 530, 48, 18, "HOG"));

	textAreaList.push_back(TextArea(270, 470, 148, 18, "Active Model Shape"));

	textAreaList.push_back(TextArea(510, 470, 48, 18, "SVM"));
	textAreaList.push_back(TextArea(510, 500, 48, 18, "DNN"));
	textAreaList.push_back(TextArea(510, 530, 48, 18, "CNN"));

	textAreaList.push_back(TextArea(680, 470, 58, 18, "Image"));

	imageFrameList.push_back(ImageFrame(20, 60));
	imageFrameList.push_back(ImageFrame(450, 60));

	checkBoxList.push_back(CheckBox(46, 475));
	checkBoxList.push_back(CheckBox(46, 505));
	checkBoxList.push_back(CheckBox(46, 535));

	checkBoxList.push_back(CheckBox(246, 475));

	checkBoxList.push_back(CheckBox(486, 475));
	checkBoxList.push_back(CheckBox(486, 505));
	checkBoxList.push_back(CheckBox(486, 535));

	checkBoxList.push_back(CheckBox(666, 475));

	inputAreaList.push_back(InputArea(100, 360, 658, 25, ""));

	checkBoxList[0].switchTexture();
	checkBoxList[3].switchTexture();
	checkBoxList[4].switchTexture();
	checkBoxList[7].switchTexture();



	this->numberOfFaces = 0;
	this->currentFace = 0;
	this->displayMode = 0;
	this->errorMessage = "";
}

Application::~Application()
{
}

void Application::run()
{
	while (window.isOpen())
	{
		window.clear(sf::Color::White);
		draw();
		window.display();
		pollEvent();
	}
}

void Application::draw()
{
	sf::RectangleShape background;
	background.setPosition(sf::Vector2f(0.0f, 0.0f));
	background.setSize(sf::Vector2f(800.0f, 600.0f));
	background.setFillColor(sf::Color(166, 166, 166));
	window.draw(background);
	for (int i = 0; i < classicButtonList.size(); i++)
	{
		classicButtonList[i].render(&window, font);
	}
	for (int i = 0; i < buttonList.size(); i++)
	{
		buttonList[i].render(&window, font);
	}
	
	for (int i = 0; i < textAreaList.size(); i++)
	{
		textAreaList[i].render(&window, font);
	}
	for (int i = 0; i < imageFrameList.size(); i++)
	{
		imageFrameList[i].render(&window);
	}
	for (int i = 0; i < checkBoxList.size(); i++)
	{
		checkBoxList[i].render(&window);
	}
	for (int i = 0; i < inputAreaList.size(); i++)
	{
		inputAreaList[i].render(&window, font);
	}
	errorBox.setText(errorMessage);
	errorBox.render(&window, font);
}

void Application::update()
{
}

void Application::pollEvent()
{
	sf::Event e;
	sf::Vector2i mousePos = sf::Mouse::getPosition(window);

	while (window.pollEvent(e))
	{
		if (e.type == sf::Event::Closed)
		{
			window.close();
		}
		if (e.type == sf::Event::MouseButtonPressed)
		{
			if (e.key.code == sf::Mouse::Left)
			{
				handleMouseClick(window.mapPixelToCoords(mousePos));
			}
		}
		if (e.type == sf::Event::TextEntered)
		{
			for (int i = 0; i < inputAreaList.size(); i++)
			{
				if (inputAreaList[i].isActive() && ((e.text.unicode > 31 && e.text.unicode < 128) || e.text.unicode == 8))
				{
					if (e.text.unicode == 8)
					{
						inputAreaList[i].deleteLetter();
					}
					else
					{ 
						inputAreaList[i].addLetter(static_cast<char>(e.text.unicode));
					}
					break;
				}
			}
		}
		if (e.type == sf::Event::KeyPressed)
		{
			for (int i = 0; i < inputAreaList.size(); i++)
			{
				if (inputAreaList[i].isActive())
				{
					if (e.key.control && e.key.code == sf::Keyboard::V)
					{
						std::string path = sf::Clipboard::getString();
						inputAreaList[i].setPath(path);
					}
					break;
				}
			}
		}
	}

}

void Application::handleMouseClick(const sf::Vector2f & mousePos)
{
	for (int i = 0; i < inputAreaList.size(); i++)
	{
		if (inputAreaList[i].isActive())
		{
			inputAreaList[i].switchActive();
		}
		if (inputAreaList[i].isPressed(mousePos))
		{
			inputAreaList[i].switchActive();
		}
	}

	for (int i = 0; i < checkBoxList.size(); i++)
	{
		if (checkBoxList[i].isPressed(mousePos))
		{
			if (i == 0 || i < 3)
			{
				for (int j = 0; j < 3; j++)
				{
					if (checkBoxList[j].isActive())
					{
						checkBoxList[j].switchTexture();
					}
				}
				checkBoxList[i].switchTexture();
				break;
			}
			else if (i == 3)
				break;
			else if (i > 3 && i < 7)
			{
				for (int j = 4; j <= 6; j++)
				{
					if (checkBoxList[j].isActive())
					{
						checkBoxList[j].switchTexture();
					}
				}
				checkBoxList[i].switchTexture();
				break;
			}
			else if (i==7)
			{
				break;
			}
		}
	}
	for (int i = 0; i < classicButtonList.size(); i++)
	{
		if (classicButtonList[i].isPressed(mousePos))
		{
			connector.resetPredictor();
			imageFrameList[0].resetFrame();
			imageFrameList[1].resetFrame();
			if (inputAreaList[0].getPath().size() == 0)
			{
				this->errorMessage = "Path cant be empty";
				break;
			}
			if(!imageFrameList[0].setImage(inputAreaList[0].getPath()))
			{
				this->errorMessage = "This file doesn't exist or extension isnt supported";
				break;
			}
			else
			{
				this->errorMessage = "";
				std::string imagePath = inputAreaList[0].getPath();
				int faceDetector = -1;
				int classificator = -1;
				if (checkBoxList[0].isActive())
				{
					faceDetector = 1;
				}
				else if (checkBoxList[1].isActive())
				{
					faceDetector = 2;
				}
				else
				{
					faceDetector = 3;
				}

				if (checkBoxList[4].isActive())
				{
					classificator = 1;
				}
				else if (checkBoxList[5].isActive())
				{
					classificator = 2;
				}
				else
				{
					classificator = 3;
				}
				int response = connector.detectEmotions(imagePath, faceDetector, classificator);
				if (response == -1)
				{
					this->errorMessage = "No face on image";
					break;
				}
				this->errorMessage = "";
				this->numberOfFaces = (int)connector.getNumberOfFaces();
				imageFrameList[1].setImage("results/allEmotions.jpg");
			}
		}
	}
	if (errorMessage.size() != 0)
	{
		return;
	}
	for (int i = 0; i < buttonList.size(); i++)
	{
		if (buttonList[i].isPressed(mousePos))
		{
			switch (i)
			{
			case 0:
				displayMode++;
				if (displayMode > 3)
					displayMode = 0;
				break;
			case 1:
				displayMode--;
				if (displayMode < 0)
					displayMode = 3;
				break;
			case 2:
				currentFace--;
				if (currentFace < 0)
				{
					currentFace = numberOfFaces - 1;
					if (currentFace < 0)
						currentFace = 0;
				}
				break;
			case 3:
				currentFace++;
				if (currentFace >= numberOfFaces)
					currentFace = 0;
				break;
			}

			switch (displayMode)
			{
			case 0:
				imageFrameList[1].setImage("results/allEmotions.jpg");
				break;
			case 1:
				imageFrameList[1].setImage("results/allFeatures.jpg");
				break;
			case 2:
				connector.getFace(this->currentFace);
				imageFrameList[1].setImage("results/emotionFace.jpg");
				break;
			case 3:
				connector.getFeatures(this->currentFace);
				imageFrameList[1].setImage("results/featureFace.jpg");
				break;
			}
		}
	}
}
