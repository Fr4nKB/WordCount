#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

#include <time.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iterator>

using namespace std;

int NBLOCK = 1, NTHREAD = 1, MAX_ELEM = 1;

__global__ void initRNG(curandState* const rngStates, const unsigned int seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &rngStates[tid]);
}


__global__ void init_array(int *words_int, int *totlen, int *var, curandState* const rngStates) {
    int end = *totlen;
    int range = *var;

    for(int i = 0; i < end; i++) {
        words_int[i] = int(curand(&rngStates[0])%range);
    }

}

__global__ void thread_func(int *words_int, int *nwords, int *var, int *result, int *len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wordsPerThread = *nwords;
    int start = tid*wordsPerThread;
    int size = *len;
    int end = (tid+1)*wordsPerThread;
    int range = *var;
    int parent = tid>>5;

    for(int i = start; i < end; i++) {
        if(i < size) atomicAdd(&result[parent*range+words_int[i]], 1);
    }

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
vector<double> mainthread(int *d_words, int *totlen, int *d_wordsXthread, int *var) {

    int *h_results = new int[MAX_ELEM * NBLOCK * NTHREAD/32], *d_results;
    unordered_map<int, int>res;

    cudaMalloc((void**)&d_results, sizeof(int) * MAX_ELEM * NBLOCK * NTHREAD/32);
    cudaMemset(d_results, 0, sizeof(int) * MAX_ELEM * NBLOCK * NTHREAD/32);
    
    clock_t startTime = clock();

    thread_func<<<NBLOCK, NTHREAD>>>(d_words, d_wordsXthread, var, d_results, totlen);
	cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, sizeof(int) * MAX_ELEM * NBLOCK * NTHREAD/32, cudaMemcpyDeviceToHost);

    //merge results from threads
    for(int i = 0; i < MAX_ELEM; i++) {
        for(int j = 0; j < NBLOCK * NTHREAD/32; j++) {
            res[i] += h_results[j*MAX_ELEM+i];
        }
    }

    clock_t endTime = clock();

    delete[] h_results;
    cudaFree(d_results);

    vector<double> times;
    times.push_back(0);
    times.push_back((endTime - startTime)/1000.0);

    return times;

}

void benchmark(string n, int totlen, int variabilty, int warps, int n_iter) {
    int h_wordsXthread, *d_wordsXthread, *d_words, *var, *array_len;
    vector<vector<double>>avg_res;
    vector<double>avg(2);
	curandState* devStates;

    MAX_ELEM = variabilty;

    cudaMallocHost((void**)&d_words, sizeof(int) * totlen);
    cudaMalloc((void**)&array_len, sizeof(int));
    cudaMalloc((void**)&d_wordsXthread, sizeof(int));
    cudaMalloc((void**)&var, sizeof(int));
	cudaMalloc((void**)&devStates, sizeof(curandState));

    cudaMemcpy(array_len, &totlen, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wordsXthread, &h_wordsXthread, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(var, &MAX_ELEM, sizeof(int), cudaMemcpyHostToDevice);

    initRNG<<<1, 1>>>(devStates, time(NULL));
    init_array<<<1, 1>>>(d_words, array_len, var, devStates);

    for(int i = 0; NTHREAD < warps*32; i++) {

        NTHREAD = 32*pow(2,i);
        h_wordsXthread = ceil(((float) totlen)/(float)(NBLOCK * NTHREAD));
        cudaMemcpy(d_wordsXthread, &h_wordsXthread, sizeof(int), cudaMemcpyHostToDevice);
        avg[0] = avg[1] = 0;
        
        cout<<"STARTING THREADS ("<<NBLOCK<<", "<<NTHREAD<<")"<<endl;
        for(int j = 0; j < n_iter; j++) {
            cout<<"Run: "<<j+1<<endl;
            auto tmp = mainthread(d_words, array_len, d_wordsXthread, var);
            avg[0] += tmp[0]/n_iter; avg[1] += tmp[1]/n_iter;
        }
        avg_res.push_back(avg);

    }

    cudaFree(d_words);
    cudaFree(d_wordsXthread);
    cudaFree(var);
    cudaFree(devStates);

    fstream output;
    output.open("output_v2_b"+to_string(NBLOCK)+"_n_"+to_string(totlen)+".txt", ios::out);
    for(int i = 0; i < avg_res.size(); i++) {
        output<<avg_res[i][0]<<"\t\t"<<avg_res[i][1]<<endl;
    }
    output.close();

}

// arrayDim maxElem nblocks nthreads n_iter
int main(int argc, char* argv[]) {

    if(argc != 6) return -1;
    NBLOCK = stoi(argv[3]);
    benchmark("", stoi(argv[1]), stoi(argv[2]), stoi(argv[4]), stoi(argv[5]));

    return 0;
}