BINARY_NAME = cks
CUDA_PATH   = /usr/local/cuda-10.1
CC			= $(CUDA_PATH)/bin/nvcc
CFLAGS		= -O3 -std=c++11
LDFLAGS		= -L$(CUDA_PATH)/lib64 -lcudart -lcublas
INCFLAGS	= -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -I/include

SRC = $(wildcard src/*.cu)
OBJ = $(patsubst %.cu,%.o,$(SRC))

.PHONY : build, clean

build : $(BINARY_NAME)

$(BINARY_NAME): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) -o $(BINARY_NAME) $(OBJ)

$(OBJ) : %.o : %.cu
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) -c $< -o $@

clean:
	-rm $(BINARY_NAME)
	-rm -r src/*.o