CC ?= clang
CFLAGS ?= -shared -fPIC -O3 -Wall -Wno-unused-result
INCLUDES = -I include -I third_party/onnxruntime/include

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIB_NAME = liblib_onnx_bridge.dylib
	LDFLAGS = -L third_party/onnxruntime/lib/macos_arm64 -lonnxruntime \
	          -install_name @rpath/$(LIB_NAME) \
	          -Wl,-rpath,@loader_path/../third_party/onnxruntime/lib/macos_arm64 \
	          -Wl,-rpath,@loader_path/.
else
	LIB_NAME = liblib_onnx_bridge.so
	LDFLAGS = -L third_party/onnxruntime/lib -lonnxruntime -Wl,-rpath,'$$ORIGIN'
endif

BUILD_DIR = build
TARGET = $(BUILD_DIR)/$(LIB_NAME)

all: $(TARGET)

$(TARGET): src/lib_onnx_bridge.c include/lib_onnx.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
