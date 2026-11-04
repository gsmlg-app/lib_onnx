CC ?= clang
CFLAGS ?= -shared -fPIC -O2 -Wall -Wextra -std=c11
INCLUDES = -I include -I third_party/onnxruntime/include
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
CFLAGS += -mmacosx-version-min=14.0
ifeq ($(UNAME_M),arm64)
ORT_LIBRARY ?= third_party/onnxruntime/lib/macos_arm64/libonnxruntime.dylib
else
$(error Set an explicit target-specific ORT_LIBRARY; no pinned macOS x64 artifact)
endif
LIB_NAME = liblib_onnx_bridge.dylib
LDFLAGS ?= $(ORT_LIBRARY) -pthread -install_name @rpath/$(LIB_NAME) -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../third_party/onnxruntime/lib/macos_arm64
else
ORT_LIBRARY ?= third_party/onnxruntime/lib/libonnxruntime.so.1
LIB_NAME = liblib_onnx_bridge.so
LDFLAGS ?= $(ORT_LIBRARY) -pthread -ldl -Wl,-rpath,'$$ORIGIN'
endif
BUILD_DIR ?= build
TARGET = $(BUILD_DIR)/$(LIB_NAME)
SOURCES = src/lib_onnx_bridge.c src/library_locator.c

all: $(TARGET)
$(TARGET): $(SOURCES) include/lib_onnx.h $(ORT_LIBRARY)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) $(LDFLAGS) -o $@

.PHONY: all
