PREFIX:=~/opt/dev/llvm-project/destdir

all:
	cmake -G Ninja -B./build . -DMLIR_DIR=$(PREFIX)/lib/cmake/mlir
	ninja -C ./build

run:
	./build/muc ./test/test2.mu

clean:
	rm -rf build
