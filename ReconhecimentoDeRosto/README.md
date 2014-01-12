

Trabalho da disciplina Engenharia de Sistemas Ubíquos
-----------------------------------------------------

Este projeto consiste na utilização da visão computacional para
reconhecer rostos baseados nas fotos contidas em uma base, foi
utilizado o OpenCv como back-end de processamento dos padrões das
imagens para então reconhecer estes padrões nas imagens da câmera.

No Makefile do projeto existe o comando que faz o projeto compilar,
se estiver utilizando uma arquitetura de 64 bits utilize o comando abaixo:

g++ *.cpp `pkg-config --cflags --libs opencv` -o release/reconhecimentoDeRosto -m64 -arch x86_64

se estiver em uma arquitetura de 32 bits então utilize:

g++ *.cpp `pkg-config --cflags --libs opencv` -o release/reconhecimentoDeRosto -m32 -arch i386

Este projeto foi promovido pela disciplina Engenharia de Sistemas
Ubíquos do curso de Especialização do IFBA em Computação Distribuída
e Ubíqua no GSORT acompanhado pelo professor Manoel Neto. 