all:
	g++ src/*.cpp `pkg-config --cflags --libs opencv` -o release/reconhecimentoDeRosto -m64 -arch x86_64
