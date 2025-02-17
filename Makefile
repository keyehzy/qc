CXX = g++
CXXFLAGS = -Wall -Wextra -g -O2 -std=c++20
LDFLAGS = -lgsl -lgslcblas -lm

TARGET = main

SOURCES = main.cpp orbital.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
