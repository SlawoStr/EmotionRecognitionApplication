#include "Application.h"

Application::Application() : window(sf::VideoMode(800, 600), "Emotion Recognition Application", sf::Style::Close | sf::Style::Titlebar)
{
	window.setFramerateLimit(60);
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
}

void Application::update()
{
}

void Application::pollEvent()
{
	sf::Event e;

	while (window.pollEvent(e))
	{
		switch (e.type)
		{
			case sf::Event::Closed:
			{			
				window.close();
				break;
			}
			case sf::Event::MouseButtonPressed:
			{
				sf::Vector2i mousePos = sf::Mouse::getPosition(window);
				pyConnector.testFunction();
				break;
			}
		}
	}
}
