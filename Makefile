GXX := g++
LIB := -lcudnn -lcudart
INC := /usr/local/cuda-9.0/include
LIBRARY= /usr/local/cuda-9.0/lib64

all:
	$(GXX) cudnntest.cpp  -I$(INC) -L$(LIBRARY) $(LIB)
