CC=/opt/homebrew/opt/llvm/bin/clang
CXX=/opt/homebrew/opt/llvm/bin/clang++
CXXFLAGS = -Wall -Wextra -g -O2  -fno-omit-frame-pointer -fsanitize=address,undefined -std=c++20 -I/opt/homebrew/Cellar/gsl/2.8/include -I/opt/homebrew/opt/openblas/include -I/opt/homebrew/Cellar/eigen/3.4.0_1/include -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I/opt/homebrew/Cellar/libomp/21.1.2/include -fopenmp
LDFLAGS = -lgsl -lgslcblas -lm -L/opt/homebrew/Cellar/gsl/2.8/lib -L/opt/homebrew/opt/openblas/lib -lopenblas -L/opt/homebrew/Cellar/libomp/21.1.2/lib -fopenmp

TARGET = main

SOURCES = src/main.cpp src/orbital.cpp src/basis_set.cpp src/hartree_fock.cpp src/lda.cpp src/uniform_grid.cpp src/atom_centered_grid.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
