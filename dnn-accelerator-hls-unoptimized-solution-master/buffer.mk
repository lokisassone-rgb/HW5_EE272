CATAPULT = /cad/mentor/2019.11/Catapult_Synthesis_10.4b-841621/Mgc_home/bin/catapult
MGC_HOME = /cad/mentor/2019.11/Catapult_Synthesis_10.4b-841621/Mgc_home
CC = $(MGC_HOME)/bin/g++
CFLAGS += -g -std=c++11

run_weight_tb: weight_tb
	./weight_tb

weight_tb: ../src/WeightDoubleBufferTb.cpp 
	$(CC) $(CFLAGS) -I$(MGC_HOME)/shared/include -I../src ../src/WeightDoubleBufferTb.cpp -o $@

run_input_tb: input_tb
	./input_tb

input_tb: ../src/InputDoubleBufferTb.cpp 
	$(CC) $(CFLAGS) -I$(MGC_HOME)/shared/include -I../src ../src/InputDoubleBufferTb.cpp -o $@

run_conv_tb: conv_tb
	./conv_tb

conv_tb: ../src/Conv.cpp ../src/ConvTb.cpp 
	$(CC) $(CFLAGS) -I$(MGC_HOME)/shared/include -I../src ../src/Conv.cpp ../src/ConvTb.cpp -o $@

.PHONY: clean
clean:
	rm -f weight_tb
	rm -f input_tb
	rm -f conv_tb
