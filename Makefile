all:
	g++ -c -O3 common.cc -o common.o
	nvcc -c receive.cu receive.o 
	nvcc -o receive receive.o common.o
	g++ -c send.cc -o send.o
	g++ -o send send.o common.o
	rm common.o receive.o send.o

clean:
	rm common.o receive.o receive send
