# Makefile
SOURCES =*.cpp
TARGET= main
CXX_FLAGS=-std=c++11 -lOpenCL


all: 
	g++ $(SOURCES) $(CXX_FLAGS) -o $(TARGET)

.PHONY: clean

clean:
	rm -vf $( TARGET)