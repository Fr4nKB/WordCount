#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <unordered_map>
#include <queue>

using namespace std;

//struct to count each word
typedef struct {
    string word;
    unsigned int count = 0;
} elem;

//struct for thread's arguments
typedef struct {
    bool leaf;
    int start, end;
    pthread_mutex_t mutexL, mutexR;
    bool finishedL = false, finishedR = false;
    unordered_map<string, int> lm, rm;
} args;

vector<string>words;
vector<pthread_t>threads;
vector<args>arg;
int nwords;
float alpha;

void* thread_func(void* argument) {

    int index = *((int*)argument);
    bool leaf = arg[index].leaf;
    int start = arg[index].start;
    int end = arg[index].end;
    int parent = floor((index-1)/2);
    int count, obj = floor(nwords*alpha);
    char child = (2*parent+1 == index) ? 'l' : 'r';
    bool statusL = false, statusR = false;
    unordered_map<string, int> m;
    unordered_map<string, int>::iterator it;
    
    if(leaf == true) {
        count = 0;
        for(int i = start; i < end; i++) {

            string word = words[i];

            if(m.find(word) == m.end()) m.insert({word, 1});
            else {
                it = m.find(word);
                it->second += 1;
            }
            count++;
            if(parent >= 0 && index != 0 && (count > obj || i == end - 1)) {
                count = 0;
                if(child == 'l') {
                    pthread_mutex_lock(&(arg[parent].mutexL));
                    arg[parent].lm.insert(m.begin(), m.end());
                    if(i < end - 1) arg[parent].finishedL = false;
                    else arg[parent].finishedL = true;
                    pthread_mutex_unlock(&(arg[parent].mutexL));
                }
                else {
                    pthread_mutex_lock(&(arg[parent].mutexR));
                    arg[parent].rm.insert(m.begin(), m.end());
                    if(i < end - 1) arg[parent].finishedR = false;
                    else arg[parent].finishedR = true;
                    pthread_mutex_unlock(&(arg[parent].mutexR));
                }
                m.clear();
            }
        }
    }

    else {
        while(!(statusL && statusR)) {

            pthread_mutex_lock(&(arg[index].mutexL));
            statusL = arg[index].finishedL;
            for(it = arg[index].lm.begin(); it != arg[index].lm.end(); it++) m[it->first] += it->second;
            if(arg[index].lm.size() > 0) arg[index].lm.clear();
            pthread_mutex_unlock(&(arg[index].mutexL));

            pthread_mutex_lock(&(arg[index].mutexR));
            statusR = arg[index].finishedR;
            for(it = arg[index].rm.begin(); it != arg[index].rm.end(); it++) m[it->first] += it->second;
            if(arg[index].rm.size() > 0) arg[index].rm.clear();
            pthread_mutex_unlock(&(arg[index].mutexR));
            
            if(parent >= 0 && index != 0) {
                if(child == 'l') {
                    pthread_mutex_lock(&(arg[parent].mutexL));
                    arg[parent].lm.insert(m.begin(), m.end());
                    arg[parent].finishedL = statusL && statusR;
                    pthread_mutex_unlock(&(arg[parent].mutexL));
                }
                else {
                    pthread_mutex_lock(&(arg[parent].mutexR));
                    arg[parent].rm.insert(m.begin(), m.end());
                    arg[parent].finishedR = statusL && statusR;
                    pthread_mutex_unlock(&(arg[parent].mutexR));
                }
                m.clear();
            }
        }
    }

    return NULL;

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
double mainthread(int height) {
    int Nthreads = pow(2, height+1) - 1;
    int Nleafs = pow(2, height);
    threads.clear();
    arg.clear();
    threads.resize(Nthreads);
    arg.resize(Nthreads);
    unordered_map<string, int> result;
    unordered_map<string, int>::iterator it;
    int totlen, main_res_len, thread_res_len, index;
    int tid[Nthreads];
    
    totlen = words.size();
    nwords = int(totlen/Nleafs);
    alpha = (height == 0) ? 1 : 1.0/Nleafs;

    for(int i = 0; i < Nthreads; i++) {

        tid[i] = i;

        if(i < Nthreads - Nleafs) arg[i].leaf = false;
        else {
            arg[i].leaf = true;
            arg[i].start = (i - (Nthreads - Nleafs))*nwords;
            arg[i].end = (i == Nthreads - 1) ? totlen : (i - (Nthreads - Nleafs) + 1)*nwords;
            pthread_mutex_init(&(arg[i].mutexL), NULL);
            pthread_mutex_init(&(arg[i].mutexR), NULL);
        }
        
    }

    clock_t startTime = clock();

    for(int i = Nthreads - 1; i >= 0; i--) {
        pthread_create(&threads[i], NULL, thread_func, (void*)&tid[i]);
    }

    pthread_join(threads[0], NULL);

    return (clock() - startTime)/1000.0;

}

void benchmark(string n, string filename, int height, int n_iter) {

    vector<double>avg;

    //copies all the words in 'input.txt' inside a vector
    clock_t startTime = clock();
    ifstream file(filename.c_str());
    copy(istream_iterator<string>(file), istream_iterator<string>(), back_inserter(words));
    double fileLoadTime = (clock() - startTime)/1000.0;
    cout<<"Time to load file: "<<fileLoadTime<<" s"<<endl;

    for(int i = 0; i <= height; i++) {
        
        avg.push_back(0);

        for(int j = 0; j < n_iter; j++) {
            cout<<"Height: "<<i<<" Run: "<<j+1<<endl;
            avg[i] += ((mainthread(i))/n_iter);
        }

        cout<<"AVG runtime: "<<avg[i]<<" s"<<endl<<endl;

    }

    //avg run times saved to 'output.txt'
    fstream output;
    output.open("output_"+n+"_t.txt", ios::out);
    for(int i = 0; i < avg.size(); i++) {
        output<<avg[i]<<endl;
    }
    output.close();

}

int main(int argc, char* argv[]) {

    if(argc != 4) return -1;

    int height = stoi(argv[1]);
    int n_iter = stoi(argv[2]);
    vector<string>dim = {"1", "10", "100", "1000"};
    
    for(int i = 0; i < dim.size(); i++) {
        system(("gen "+dim[i]+" "+argv[3]).c_str());
        benchmark(dim[i], "input.txt", height, n_iter);
        system(("python graph.py "+dim[i]+"_t").c_str());
    }

    //average case, moby dick
    benchmark("mobyDick", "mobyDick.txt", height, n_iter);
    system("python graph.py mobyDick_t");

    //average case, collection of books
    benchmark("collection", "collection.txt", height, n_iter);
    system("python graph.py collection_t");

    return 0;
}