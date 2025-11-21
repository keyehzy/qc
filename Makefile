CXX = g++
CXXFLAGS = -Wall -Wextra -g -O2 -std=c++20 -I/opt/homebrew/Cellar/gsl/2.8/include -I/opt/homebrew/Cellar/eigen/3.4.0_1/include
LDFLAGS = -lgsl -lgslcblas -lm -L/opt/homebrew/Cellar/gsl/2.8/lib

TARGET = main

SOURCES = src/main.cpp src/orbital.cpp src/basis_set.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
