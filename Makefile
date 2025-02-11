CXX = g++
CXXFLAGS = -Wall -Wextra -g -std=c++20

TARGET = main

SOURCES = main.cpp orbital.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
