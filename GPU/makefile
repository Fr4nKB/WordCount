#test to compare all the versions togheter
#due to the first version being memory hungry the data set has to be limited
CUDAPARAMS_16_1M_32K = 1000000 32768 16 32 20
CUDAPARAMS_32_1M_32K = 1000000 32768 32 32 20
CUDAPARAMS_68_1M_32K = 1000000 32768 68 32 20

#test to compare with CPU with the 100MB file (collection of books)
CUDAPARAMS_VS_CPU_1 = 18732150 126153 1 32 20
CUDAPARAMS_VS_CPU_4 = 18732150 126153 4 32 20
CUDAPARAMS_VS_CPU_16 = 18732150 126153 16 32 20
CUDAPARAMS_VS_CPU_34 = 18732150 126153 34 32 20
CUDAPARAMS_VS_CPU_68 = 18732150 126153 68 32 20

#test with 4gb of data
CUDAPARAMS_1_500M_50M = 500000000 50000000 1 32 20
CUDAPARAMS_4_500M_50M = 500000000 50000000 4 32 20
CUDAPARAMS_16_500M_50M = 500000000 50000000 16 32 20
CUDAPARAMS_34_500M_50M = 500000000 50000000 34 32 20
CUDAPARAMS_68_500M_50M = 500000000 50000000 68 32 20

run: build
	
	./wc1 $(CUDAPARAMS_16_1M_32K)
	./wc1 $(CUDAPARAMS_32_1M_32K)
	
	./wc2 $(CUDAPARAMS_16_1M_32K)
	./wc2 $(CUDAPARAMS_32_1M_32K)

	./wcI $(CUDAPARAMS_16_1M_32K)
	./wcI $(CUDAPARAMS_32_1M_32K)
	./wcI $(CUDAPARAMS_68_1M_32K)

	./wc3 $(CUDAPARAMS_16_1M_32K)
	./wc3 $(CUDAPARAMS_32_1M_32K)
	./wc3 $(CUDAPARAMS_68_1M_32K)

	./wc4 $(CUDAPARAMS_16_1M_32K)
	./wc4 $(CUDAPARAMS_32_1M_32K)
	./wc4 $(CUDAPARAMS_68_1M_32K)

	./wc4 $(CUDAPARAMS_VS_CPU_1)
	./wc4 $(CUDAPARAMS_VS_CPU_4)
	./wc4 $(CUDAPARAMS_VS_CPU_16)
	./wc4 $(CUDAPARAMS_VS_CPU_34)
	./wc4 $(CUDAPARAMS_VS_CPU_68)

	./wc4 $(CUDAPARAMS_1_500M_50M)
	./wc4 $(CUDAPARAMS_4_500M_50M)
	./wc4 $(CUDAPARAMS_16_500M_50M)
	./wc4 $(CUDAPARAMS_34_500M_50M)
	./wc4 $(CUDAPARAMS_68_500M_50M)

build: wc1 wcI wc2 wc3 wc4

wc1: WordCount_V1.cu
	nvcc WordCount_V1.cu -o wc1

wcI: WordCount_VI.cu
	nvcc WordCount_VI.cu -o wcI

wc2: WordCount_V2.cu
	nvcc WordCount_V2.cu -o wc2

wc3: WordCount_V3.cu
	nvcc WordCount_V3.cu -o wc3

wc4: WordCount_V4.cu
	nvcc WordCount_V4.cu -o wc4

clean:
	del output*.txt *.exe *.lib *.exp