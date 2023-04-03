#include "AppUtilities.h"

ClassicButton::ClassicButton(float x, float y, float width, float height, std::string text, sf::Color color)
{
	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setSize(sf::Vector2f(width, height));

	this->text.setString(text);
	this->text.setFillColor(sf::Color::Black);
	this->text.setCharacterSize(12);

	this->buttonColor = color;
	this->shape.setFillColor(buttonColor);
	this->shape.setOutlineColor(sf::Color::Black);
	this->shape.setOutlineThickness(1.0f);
}

ClassicButton::~ClassicButton()
{
}

void ClassicButton::render(sf::RenderTarget * target, const sf::Font & font)
{
	target->draw(this->shape);

	this->text.setFont(font);
	this->text.setPosition(
		this->shape.getPosition().x + (this->shape.getGlobalBounds().width / 2.f) - this->text.getGlobalBounds().width / 2.f,
		this->shape.getPosition().y + (this->shape.getGlobalBounds().height / 2.f) - this->text.getGlobalBounds().height / 2.f
	);
	target->draw(this->text);
}

bool ClassicButton::isPressed(const sf::Vector2f & mousePos)
{
	if (this->shape.getGlobalBounds().contains(mousePos))
	{
		return true;
	}
	return false;
}

Button::Button(float x, float y, std::string path)
{
	if (!texture.loadFromFile(path))
	{
		std::cout << "Fail" << std::endl;
	}
	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setScale(0.5, 0.5);	
}

Button::~Button()
{
}

void Button::render(sf::RenderTarget * target, const sf::Font & font)
{
	this->shape.setTexture(texture);
	target->draw(this->shape);
}

bool Button::isPressed(const sf::Vector2f & mousePos)
{
	if (this->shape.getGlobalBounds().contains(mousePos))
	{
		return true;
	}
	return false;
}

TextArea::TextArea(float x, float y, float width, float height, std::string text)
{
	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setSize(sf::Vector2f(width, height));

	this->text.setString(text);
	this->text.setFillColor(sf::Color::Black);
	this->text.setCharacterSize(12);
}

TextArea::~TextArea()
{
}

void TextArea::render(sf::RenderTarget * target, const sf::Font & font)
{
	this->text.setFont(font);
	this->text.setPosition(
		this->shape.getPosition().x + (this->shape.getGlobalBounds().width / 2.f) - this->text.getGlobalBounds().width / 2.f,
		this->shape.getPosition().y + (this->shape.getGlobalBounds().height / 2.f) - this->text.getGlobalBounds().height / 2.f
	);
	target->draw(this->text);
}

void TextArea::setText(const std::string & errormsg)
{
	this->text.setString(errormsg);
}

ImageFrame::ImageFrame(float x, float y)
{
	int width = 260;
	int height = 260;
	sf::Uint8* pixels = new sf::Uint8[width * height * 4];
	for (int i = 0; i < width*height * 4; i ++)
	{
		pixels[i] = 255;
	}
	texture.create(260, 260);
	texture.update(pixels);
	this->shape.setPosition(sf::Vector2f(x, y));
}

ImageFrame::~ImageFrame()
{
}

void ImageFrame::render(sf::RenderTarget * target)
{
	this->shape.setTexture(texture);
	this->shape.setScale(260.0f / shape.getLocalBounds().width, 260.0f / shape.getLocalBounds().height);
	target->draw(this->shape);
}

bool ImageFrame::setImage(std::string imagePath)
{
	if (!texture.loadFromFile(imagePath))
	{
		return false;
	}
	this->shape.setTextureRect(sf::IntRect(sf::Vector2i(0, 0), sf::Vector2i(this->shape.getTexture()->getSize())));

	return true;
}

void ImageFrame::resetFrame()
{
	int width = 260;
	int height = 260;
	sf::Uint8* pixels = new sf::Uint8[width * height * 4];
	for (int i = 0; i < width*height * 4; i++)
	{
		pixels[i] = 255;
	}
	texture.create(260, 260);
	texture.update(pixels);
}

CheckBox::CheckBox(float x, float y)
{
	this->activeTexture = 1;
	if (!texture.loadFromFile("resources/icons/checkBox.png"))
	{
		std::cout << "Fail" << std::endl;
	}
	if (!confirmTexture.loadFromFile("resources/icons/checkBoxConfirm.png"))
	{
		std::cout << "Fail" << std::endl;
	}

	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setScale(0.5, 0.5);

}

CheckBox::~CheckBox()
{
}

void CheckBox::render(sf::RenderTarget * target)
{
	activeTexture==1 ? this->shape.setTexture(texture): this->shape.setTexture(confirmTexture);
	target->draw(this->shape);
}

bool CheckBox::isPressed(const sf::Vector2f & mousePos)
{
	if (this->shape.getGlobalBounds().contains(mousePos))
	{
		return true;
	}
	return false;
}

void CheckBox::switchTexture()
{
	activeTexture = activeTexture == 1 ? 2 : 1;
}

InputArea::InputArea(float x, float y, float width, float height, std::string path)
{
	this->path = path;
	this->active = false;

	this->shape.setPosition(sf::Vector2f(x, y));
	this->shape.setSize(sf::Vector2f(width, height));
	this->shape.setFillColor(sf::Color::White);

	this->text.setString(path);
	this->text.setFillColor(sf::Color::Black);
	this->text.setCharacterSize(12);
}

InputArea::~InputArea()
{
}

void InputArea::render(sf::RenderTarget * target, const sf::Font & font)
{
	this->text.setFont(font);
	this->text.setPosition(
		this->shape.getPosition().x,this->shape.getPosition().y + (this->shape.getGlobalBounds().height / 4.f));
	target->draw(this->shape);
	target->draw(this->text);
}

bool InputArea::isPressed(const sf::Vector2f & mousePos)
{
	if (this->shape.getGlobalBounds().contains(mousePos))
	{
		return true;
	}
	return false;
}

void InputArea::addLetter(char letter)
{
	path.push_back(letter);
	this->text.setString(path);
}

void InputArea::deleteLetter()
{
	if (!path.empty())
	{
		path.pop_back();
		this->text.setString(path);
	}
}

void InputArea::setPath(std::string path)
{
	this->path = path;
	this->text.setString(path);
}
