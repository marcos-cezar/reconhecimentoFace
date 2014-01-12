

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


Instruções para compilar o projeto.
-----------------------------------

Execute o comando 

git clone https://github.com/marcos-cezar/reconhecimentoFace.git

em seguida execute 

cd ReconhecimentoDeRosto

por último execute

make

no diretório release estará o executável de nome
reconhecimentoDeRosto que deverá ser executado em linha de comando,
passando 2 arquivos que descrevem o padrão de imagem
(haarcascade_frontalface_alt.xml, haarcascade_eye_tree_eyeglasses.xml)
e o arquivo csv (faces.csv)
que descreve o caminho do conjunto de imagens para treinar o algoritmo.


