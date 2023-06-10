#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <iterator>
#include <time.h>

using namespace std;

//struct to implement a list and count each word
typedef struct elem {
    string word;
    unsigned int count = 0;
    elem *next;
} elem;


class List {
    elem* head = NULL;
    unsigned int size = 0;
public:
    List();
    void print();
    void insert(string, unsigned int);
    elem* isPresent(string, unsigned int);
    elem* find(unsigned int);
    unsigned int len() { return this->size; };
    ~List();

};

List words;

List::List() {
    this->head = NULL;
}

void List::insert(string word, unsigned int count) {
    if(!this->head) {
        this->head = new elem;
        this->head->word = word;
        this->head->count = count;
        this->head->next = NULL;
        this->size++;
        return;   
    }

    elem* tmp = new elem;
    tmp->word = word;
    tmp->count = count;
    tmp->next = this->head;
    this->head = tmp;
    this->size++;

    return;
}

void List::print() {
    elem *p = this->head;
    while(p) {
        cout<<p->word<<" "<<p->count<<endl;
        p = p->next;
    }
    return;
}

elem* List::isPresent(string word, unsigned int toAdd) {
    elem *p = this->head;

    while(p) {
        if(p->word == word) {
            if(toAdd != 0) {
                p->count += toAdd;
            }
            return p;
        }
        p = p->next;
    }

    return NULL;
}

elem* List::find(unsigned int index) {
    elem *p = this->head;
    unsigned int i = 0;
    while(p && i < index) {
        p = p->next;
        i++;
    }

    return p;
}

List::~List() {
    elem *p = this->head;
    while(this->head) {
        this->head = this->head->next;
        delete p;
        p = this->head;
    }
}

//struct for thread's arguments
typedef struct {
    unsigned int index, start, end;
    List res;
} args;

void* thread_func(void* arg) {

    args* arguments = (args*)arg;
    unsigned int index = arguments->index;
    unsigned int start = arguments->start;
    unsigned int end = arguments->end;

    elem *p = words.find(start);
    unsigned int i = start;
    while(i < end && p) {
        i++;
        if(arguments->res.isPresent(p->word, p->count) == NULL) {
            arguments->res.insert(p->word, p->count);
        }
        p = p->next;
    }

    return NULL;

}

//launches 'Nthreads' and gives to each one of them a portion of 'words'
double mainthread(int Nthreads) {
    vector<pthread_t>threads(Nthreads);
    vector<args>arg(Nthreads);
    List result;
    unsigned int totlen, nwords, main_res_len, thread_res_len, index;
    
    totlen = words.len();
    nwords = int(totlen/Nthreads);

    for(int i = 0; i < Nthreads; i++) {
        arg[i].index = i;
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

    unsigned int i;
    unsigned int end;
    elem *p;

    for(int i = 0; i < Nthreads; i++) {

        p = arg[i].res.find(0);

        while(p) {
            if(result.isPresent(p->word, p->count) == NULL) result.insert(p->word, p->count);
            p = p->next;
        }
        
    }

    return (clock() - startTime)/1000.0;

}

void benchmark(string n, string filename, int n_processes, int n_iter) {

    vector<double>avg;
    string word;

    //copies all the words in 'input.txt' inside a vector
    clock_t startTime = clock();
    ifstream file(filename.c_str());
    while(file >> word) {
        words.insert(word, 1);
    }
    double fileLoadTime = (clock() - startTime)/1000.0;
    cout<<"Time to load file: "<<fileLoadTime<<" s"<<endl;

    for(int i = 0; i < n_processes; i++) {
        
        avg.push_back(0);

        for(int j = 0; j < n_iter; j++) {
            cout<<"Processes: "<<i+1<<" Run: "<<j+1<<endl;
            avg[i] += ((mainthread(i+1))/n_iter);
        }

        cout<<"AVG runtime: "<<avg[i]<<" s"<<endl<<endl;

    }

    //avg run times saved to 'output.txt'
    fstream output;
    output.open("output_"+n+"_l.txt", ios::out);
    for(int i = 0; i < avg.size(); i++) {
        output<<avg[i]<<endl;
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
        system(("python graph.py "+dim[i]+"_l").c_str());
    }

    //average case, moby dick
    benchmark("mobyDick", "mobyDick.txt", n_processes, n_iter);
    system("python graph.py mobyDick_l");

    return 0;

}