
CC			:= gcc
NVCC		:= nvcc



TARGET		:= NNN
OBJ			:= $(wildcard $(BUILD_DIR)/*.cpp)
BUILD_DIR	:= build
SRC_DIR		:= src

CU_FILES	:= $(wildcard src/*.cu)
CU_OBJS		:= $(patsubst %.cu,%.o,$(CU_FILES))

C_FILES  	:= $(wildcard $(SRC_DIR)/*.c)

CCFlags		:= -I/usr/local/cuda/include -stdlib=libstdc++ -L/usr/local/cuda/lib -lcudart -lcuda
CudaFlags	:= --device-c -m64

all: cuda_operations.o $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CCFlags) $(C_FILES) -o $(BUILD_DIR)/$(TARGET)

cuda_operations.o : 
	$(NVCC) $(CudaFlags) src/cuda_operations.cu -o build/cuda_operations.o

withnvcc:
	$(NVCC) $(CU_FILES) $(C_FILES) -o $(BUILD_DIR)/app

clean:
	rm -f $(BUILD_DIR)/*.o