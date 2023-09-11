PREFIX:=~/opt/dev/llvm-project/destdir

all:
	cmake -G Ninja -B./build . -DMLIR_DIR=$(PREFIX)/lib/cmake/mlir
	ninja -C ./build

clean:
	rm -rf build
