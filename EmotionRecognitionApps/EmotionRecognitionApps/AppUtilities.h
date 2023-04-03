#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>

class ClassicButton
{
private:
	sf::RectangleShape shape;
	sf::Text text;
	sf::Color buttonColor;
public:
	ClassicButton(float x, float y, float width, float height, std::string text, sf::Color color);
	~ClassicButton();

	void render(sf::RenderTarget * target, const sf::Font & font);
	bool isPressed(const sf::Vector2f & mousePos);

};

class Button
{
private:
	sf::Sprite shape;
	sf::Texture texture;
public:
	Button(float x, float y, std::string path);
	~Button();

	void render(sf::RenderTarget * target, const sf::Font & font);
	bool isPressed(const sf::Vector2f & mousePos);
};

class TextArea
{
private:
	sf::RectangleShape shape;
	sf::Text text;
public:
	TextArea(float x, float y, float width, float height, std::string text);
	~TextArea();

	void render(sf::RenderTarget * target, const sf::Font & font);
	void setText(const std::string & errormsg);
};

class InputArea
{
private:
	sf::RectangleShape shape;
	sf::Text text;
	std::string path;
	bool active;
public:
	InputArea(float x, float y, float width, float height, std::string path);
	~InputArea();

	void render(sf::RenderTarget * target, const sf::Font & font);
	bool isPressed(const sf::Vector2f & mousePos);
	bool isActive() { return active == true ? true : false; }
	void switchActive() { active = active == true ? false : true; }
	void addLetter(char letter);
	void deleteLetter();
	void setPath(std::string path);
	std::string getPath() { return path; }
};


class ImageFrame
{
private:
	sf::Sprite shape;
	sf::Texture texture;
public:
	ImageFrame(float x, float y);
	~ImageFrame();
	
	void render(sf::RenderTarget * target);
	bool setImage(std::string imagePath);
	void resetFrame();
};

class CheckBox
{
private:
	sf::Sprite shape;
	sf::Texture texture;
	sf::Texture confirmTexture;
	int activeTexture;
public:
	CheckBox(float x, float y);
	~CheckBox();

	void render(sf::RenderTarget * target);
	bool isPressed(const sf::Vector2f & mousePos);
	void switchTexture();
	bool isActive() { if (activeTexture == 1) return false; return true; }
};