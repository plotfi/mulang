BUILD_DIR:=$(LLVM_DIR)/build
PREFIX:=$(LLVM_DIR)/destdir

DEBUG_BUILD_DIR:=$(LLVM_DIR)/debug
DEBUG_PREFIX:=$(LLVM_DIR)/destdebugdir

all:
	cmake -G Ninja -B./build . -DMLIR_DIR=$(PREFIX)/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$(BUILD_DIR)/bin/llvm-lit
	ninja -C ./build
	ln -s ./build/compile_commands.json

debug:
	cmake -G Ninja -B./build . -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR=$(DEBUG_PREFIX)/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$(DEBUG_BUILD_DIR)/bin/llvm-lit
	ninja -C ./build
	ln -s ./build/compile_commands.json



run: all
	./build/muc --emit=ast ./test/test2.mu

clean:
	rm -rf build
	rm -f compile_commands.json
