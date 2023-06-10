#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <unordered_map>

using namespace std;

vector<string>words;

//struct to count each word
typedef struct {
    string word;
    unsigned int count = 0;
} elem;

//struct for thread's arguments
typedef struct {
    vector<string>*words;
    int start, end;
    unordered_map<string, int> m;
} args;

void* thread_func(void* arg) {

    args* arguments = (args*)arg;
    int start = arguments->start;
    int end = arguments->end;

    for(int i = start; i < end; i++) {

        string word = words[i];

        if((arguments->m).find(word) == (arguments->m).end()) (arguments->m).insert({word, 1});
        else {
            unordered_map<string, int>::iterator it = (arguments->m).find(word);
            it->second += 1;
        }

    }

    return NULL;

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
vector<double> mainthread(int Nthreads) {
    vector<pthread_t>threads(Nthreads);
    vector<args>arg(Nthreads);
    vector<double>times;
    unordered_map<string, int> result;
    unordered_map<string, int>::iterator it;
    int totlen, nwords, main_res_len, thread_res_len, index;

    totlen = words.size();
    nwords = int(totlen/Nthreads);

    for(int i = 0; i < Nthreads; i++) {
        arg[i].start = i*nwords;
        arg[i].end = (i == Nthreads - 1) ? totlen : (i+1)*nwords;
    }

    clock_t startTime = clock();

    clock_t threadTime = clock();
    for(int i = 0; i < Nthreads; i++) {
        pthread_create(&threads[i], NULL, thread_func, (void*)&arg[i]);
    }

    for(int i = 0; i < Nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
    times.push_back((clock() - threadTime)/1000.0);

    clock_t mergeTime = clock();
    //merge results from threads
    result.insert(arg[0].m.begin(), arg[0].m.end());
    for(int i = 1; i < Nthreads; i++) {
        for(it = arg[i].m.begin(); it != arg[i].m.end(); it++) result[it->first] += it->second;
    }
    times.push_back((clock() - mergeTime)/1000.0);
    
    times.push_back((clock() - startTime)/1000.0);

    return times;
}

void benchmark(string n, string filename, int n_processes, int n_iter) {
    
    vector<double>times;
    vector<vector<double>>avg;
    double perc_inc = 0;

    //copies all the words in 'input.txt' inside a vector
    clock_t startTime = clock();
    ifstream file(filename.c_str());
    copy(istream_iterator<string>(file), istream_iterator<string>(), back_inserter(words));
    double fileLoadTime = (clock() - startTime)/1000.0;
    cout<<"Time to load file: "<<fileLoadTime<<" s"<<endl;

    for(int i = 0; i < n_processes; i++) {

        avg.push_back({0, 0, 0});

        for(int j = 0; j < n_iter; j++) {
            cout<<"Processes: "<<i+1<<" Run: "<<j+1<<endl;
            times = mainthread(i+1);
            for(int k = 0; k < times.size(); k++) avg[i][k] += times[k] / n_iter;
        }

        cout<<"AVG runtime: "<<avg[i][2]<<" s"<<endl<<endl;

    }

    //avg run times saved to 'output.txt'
    fstream output;
    output.open("output_"+n+"_m.txt", ios::out);
    cout<<"#T\tThreads\tMerge\tMergeInc(%)\tTotal"<<endl;
    for(int i = 0; i < avg.size(); i++) {
        cout<<i+1<<"\t"<<avg[i][0]<<"\t"<<avg[i][1]<<"\t";
        if(i == 0) cout<<"0\t";
        else {
            perc_inc = avg[i][1]*100/avg[i-1][1] - 100;
            cout<<perc_inc<<"\t";
        }
        cout<<avg[i][2]<<"\t"<<endl;
        output<<avg[i][2]<<endl;
    }
    output.close();

}

int main(int argc, char* argv[]) {

    if(argc != 4) return -1;

    int n_processes = stoi(argv[1]);
    int n_iter = stoi(argv[2]);
    vector<string>dim = {"1", "10", "100", "1000"};
    
    //worst case scenario
    for(int i = 0; i < dim.size(); i++) {
        system(("gen "+dim[i]+" "+argv[3]).c_str());
        benchmark(dim[i], "input.txt", n_processes, n_iter);
        system(("python graph.py "+dim[i]+"_m").c_str());
    }

    //average case, moby dick
    benchmark("mobyDick", "mobyDick.txt", n_processes, n_iter);
    system("python graph.py mobyDick_m");

    //average case, collection of books
    benchmark("collection", "collection.txt", n_processes, n_iter);
    system("python graph.py collection_m");

    return 0;
}