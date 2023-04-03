#pragma once
#include <SFML/Graphics.hpp>
#include "AppUtilities.h"

class WidgetManager
{
public:
	WidgetManager();

	void draw(sf::RenderWindow& window);
	void handleEvent(sf::Event e);
private:
};