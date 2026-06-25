CC=g++
NVCC=nvcc

SRC_DIR := src
OBJ_DIR := obj
LIB_OBJ_DIR := $(OBJ_DIR)/lib
BIN_DIR := bin
INS_DIR := ~/bin
AGIMUS_BIN_DIR := ../bin

SRC := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRC := $(wildcard $(SRC_DIR)/*.cu)
LIB_SRC := $(wildcard $(SRC_DIR)/lib/*.cpp)
CUDA_LIB_SRC := $(wildcard $(SRC_DIR)/lib/*.cu)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.cuda.o)
LIB_OBJ := $(LIB_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_LIB_OBJ := $(LIB_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.cuda.o)

EXECUTABLES := $(BIN_DIR)/matrixdeterminant

CPPFLAGS := -Iinclude -MMD -MP
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := -lm -lstdc++fs
NVCCFLAGS := -Iinclude

.PHONY: all clean install

all: $(EXECUTABLES)

$(BIN_DIR)/matrixdeterminant: $(OBJ_DIR)/main.cuda.o $(LIB_OBJ) $(CUDA_LIB_OBJ) | $(BIN_DIR) $(LIB_OBJ_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(LIB_OBJ_DIR)/%.o: $(SRC_DIR)/lib/%.cpp | $(LIB_OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(LIB_OBJ_DIR)/%.cuda.o: $(SRC_DIR)/lib/%.cu | $(LIB_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cuda.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BIN_DIR) $(OBJ_DIR) $(LIB_OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR)

install: 
	cp $(EXECUTABLES) $(INS_DIR)
	cp $(EXECUTABLES) $(AGIMUS_BIN_DIR)

-include $(OBJ:.o=.d)

