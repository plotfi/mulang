BUILD_DIR:=/Users/plotfi/opt/dev/llvm-project/build
PREFIX:=/Users/plotfi/opt/dev/llvm-project/destdir

all:
	cmake -G Ninja -B./build . -DMLIR_DIR=$(PREFIX)/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$(BUILD_DIR)/bin/llvm-lit
	ninja -C ./build

run: all
	./build/muc --emit=ast ./test/test2.mu

clean:
	rm -rf build
