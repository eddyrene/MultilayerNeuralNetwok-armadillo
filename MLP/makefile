all:program  

program:link_cuda.o
	g++ main.cpp -o program  link_cuda.o  -lcuda

link_cuda.o:cudaf.o
	nvcc -dlink cudaf.o -o link_cuda.o

cudaf.o:
	nvcc -c cudaf.cu

clean:
	 rm -rf *o program