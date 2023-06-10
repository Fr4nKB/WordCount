#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iterator>
#include <time.h>

using namespace std;

//struct to count each word
typedef struct {
    string word;
    unsigned int count = 0;
} elem;

//struct for thread's arguments
typedef struct {
    vector<string>*words;
    int start, end;
    vector<elem>res;
} args;

void* thread_func(void* arg) {

    args* arguments = (args*)arg;
    int start = arguments->start;
    int end = arguments->end;

    for(int i = start; i < end; i++) {

        string word = (*(arguments->words))[i];
        int index = -1;
        int len = (arguments->res).size();

        for(int j = 0; j < len; j++) {
            if((arguments->res)[j].word == word) {
                index = j;
                (arguments->res)[j].count += 1;
                break;
            }
        }

        if(index == -1) {
            elem tmp;
            tmp.word = word;
            tmp.count = 1;
            (arguments->res).push_back(tmp);
        }

    }

    return NULL;

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
double mainthread(int Nthreads, vector<string>*words) {
    vector<pthread_t>threads(Nthreads);
    vector<args>arg(Nthreads);
    vector<elem>result;
    int totlen, nwords, main_res_len, thread_res_len, index;
    
    totlen = (*words).size();
    nwords = int(totlen/Nthreads);

    for(int i = 0; i < Nthreads; i++) {
        arg[i].words = words;
        arg[i].start = i*nwords;
        arg[i].end = (i == Nthreads - 1) ? totlen : (i+1)*nwords;
    }

    clock_t startTime = clock();

    for(int i = 0; i < Nthreads; i++) {
        pthread_create(&threads[i], NULL, thread_func, (void*)&arg[i]);
    }

    for(int i = 0; i < Nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    //merge results from threads
    for(int i = 0; i < Nthreads; i++) {

        main_res_len = result.size();
        thread_res_len = (arg[i].res).size();

        for(int j = 0; j < thread_res_len; j++) {
            index = -1;

            for(int k = 0; k < main_res_len; k++) {
                
                if((arg[i].res)[j].word == result[k].word) {
                    index = k;
                    result[k].count += (arg[i].res)[j].count;
                    break;
                }
            }

            if(index == -1) {
                elem tmp;
                tmp.word = (arg[i].res)[j].word;
                tmp.count = (arg[i].res)[j].count;
                result.push_back(tmp);
            }

        }
        
    }

    return (clock() - startTime)/1000.0;

}

void benchmark(string n, string filename, int n_processes, int n_iter) {

    vector<double>avg;
    vector<string>words;

    //copies all the words in 'input.txt' inside a vector
    clock_t startTime = clock();
    ifstream file(filename.c_str());
    copy(istream_iterator<string>(file), istream_iterator<string>(), back_inserter(words));
    double fileLoadTime = (clock() - startTime)/1000.0;
    cout<<"Time to load file: "<<fileLoadTime<<" s"<<endl;

    for(int i = 0; i < n_processes; i++) {
        
        avg.push_back(0);

        for(int j = 0; j < n_iter; j++) {
            cout<<"Processes: "<<i+1<<" Run: "<<j+1<<endl;
            avg[i] += ((mainthread(i+1, &words)+fileLoadTime)/n_iter);
        }

        cout<<"AVG runtime: "<<avg[i]<<" s"<<endl<<endl;

    }

    //avg run times saved to 'output.txt'
    fstream output;
    output.open("output_"+n+"_v.txt", ios::out);
    for(int i = 0; i < avg.size(); i++) {
        output<<avg[i]<<endl;
    }
    output.close();

}

int main(int argc, char* argv[]) {

    if(argc != 4) return -1;

    unsigned int n_processes = stoi(argv[1]);
    unsigned int n_iter = stoi(argv[2]);
    vector<string>dim = {"1", "10", "100", "1000"};
    
    for(int i = 0; i < dim.size(); i++) {
        system(("gen "+dim[i]+" "+argv[3]).c_str());
        benchmark(dim[i], "input.txt", n_processes, n_iter);
        system(("python graph.py "+dim[i]+"_v").c_str());
    }

    //average case, moby dick
    benchmark("mobyDick", "mobyDick.txt", n_processes, n_iter);
    system("python graph.py mobyDick_v");

    return 0;

}