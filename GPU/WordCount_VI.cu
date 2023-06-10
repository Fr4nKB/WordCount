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


int NBLOCK = 1, NTHREAD = 32, MAX_ELEM = 1;

struct elem {
    int value;
    int counter;
};

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

__global__ void thread_func(int *words_int, int *nwords, int *totlen, elem *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_words = *nwords;
    int start = tid * thread_words;
    int end = (tid + 1) * thread_words;
    int size = *totlen;

    for (int i = start; i < end; i++) {
        if(i >= size) continue;
        for (int j = 0; j < thread_words; j++) {
            //checks if can allocate new value
            if (result[tid * thread_words + j].value == -1) {
                result[tid * thread_words + j].value = words_int[i];
                result[tid * thread_words + j].counter = 1;
                break;
            }
            //checks if same value
            if (result[tid * thread_words + j].value == words_int[i]) {
                result[tid * thread_words + j].counter += 1;
                break;
            }

            //default case so keep iterating
        }
    }

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
vector<double> mainthread(int totlen, int *d_words, int *d_wordsXthread, int *array_len, elem *d_results) {

    elem *h_results = new elem[totlen];
    unordered_map<int, int> res;
    for(int i = 0; i < totlen; i++) {
        cudaMemset(&(d_results[i].value), -1, sizeof(int));
    }
    
    clock_t startTime = clock();

    thread_func<<<NBLOCK,NTHREAD>>>(d_words, d_wordsXthread, array_len, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, sizeof(elem) * totlen, cudaMemcpyDeviceToHost);

    for(int i = 0; i < totlen; i++) {
        if(h_results[i].value != -1) res[h_results[i].value] += h_results[i].counter;
    }

    int c = 0;
    for(auto it = res.begin(); it != res.end(); it++) c += it->second;
    cout<<c<<endl;
    cout<<res.size()<<endl;

    clock_t endTime = clock();

    delete[] h_results;

    vector<double> times;
    times.push_back(0);
    times.push_back((endTime - startTime)/1000.0);

    return times;
}

void benchmark(int totlen, int variabilty, int warps, int n_iter) {
    int h_wordsXthread, *d_wordsXthread, *d_words, *var, *array_len;
    vector<vector<double>>avg_res;
    vector<double>avg(2);
    elem *d_results;
	curandState* devStates;

    MAX_ELEM = variabilty;

    cudaMallocHost((void**)&d_words, sizeof(int) * totlen);
    cudaMalloc((void**)&d_wordsXthread, sizeof(int));
    cudaMalloc((void**)&d_results, sizeof(elem) * totlen);
    cudaMalloc((void**)&var, sizeof(int));
    cudaMalloc((void**)&array_len, sizeof(int));
	cudaMalloc((void**)&devStates, sizeof(curandState));

    cudaMemcpy(var, &MAX_ELEM, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(array_len, &totlen, sizeof(int), cudaMemcpyHostToDevice);
    
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
            auto tmp = mainthread(totlen, d_words, d_wordsXthread, array_len, d_results);
            avg[0] += tmp[0]/n_iter; avg[1] += tmp[1]/n_iter;
        }
        avg_res.push_back(avg);

    }

    cudaFree(d_words);
    cudaFree(d_wordsXthread);
    cudaFree(d_results);
    cudaFree(var);
    cudaFree(devStates);

    fstream output;
    output.open("output_vI_b"+to_string(NBLOCK)+"_n_"+to_string(totlen)+".txt", ios::out);
    for(int i = 0; i < avg_res.size(); i++) {
        output<<avg_res[i][0]<<"\t\t"<<avg_res[i][1]<<endl;
    }
    output.close();

}

// arrayDim maxElem nblocks nthreads n_iter
int main(int argc, char* argv[]) {

    if(argc != 6) return -1;
    NBLOCK = stoi(argv[3]);
    benchmark(stoi(argv[1]), stoi(argv[2]), stoi(argv[4]), stoi(argv[5]));

    return 0;
}