CXX = g++
CXXFLAGS = -Wall -g -O3
#-std=gnu++98 -fPIC

LD_FLAGS = -Llib/cv -L. -lgomp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
#LD_FLAGS = -L. -lgomp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
INCLUDE_FLAGS = -Iinclude
INTRAFACE_LIB = lib/intraface/libintraface.a
#LIBFLAGS = -fopenmp

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

OBJECTS =	$(BUILD_DIR)/binary_model_file.o \
			$(BUILD_DIR)/IntrafaceTracker.o

TARGET = $(BIN_DIR)/demo

.PHONY: all clean

all: $(TARGET)
	
$(TARGET) : $(OBJECTS) $(BUILD_DIR)/IntrafaceTracker.o
	$(CXX) $(CXXFLAGS) -o  $@  $^  $(INTRAFACE_LIB) $(LD_FLAGS) 

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(INCLUDE_FLAGS) -fopenmp  -fpermissive

clean:
	$(RM) $(TARGET) $(OBJECTS)
