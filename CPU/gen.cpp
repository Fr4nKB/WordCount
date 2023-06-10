#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {  //takes in input the number of kilobytes
    if(argc != 3) return -1;
    unsigned int nB = stoi(argv[1])*1000;   //number of bytes to generate
    unsigned int choice = stoi(argv[2]);

    fstream file;
    string str = "";
    srand(time(NULL));
    file.open("input.txt", ios::out);

    unsigned int counter = 0;
    unsigned long long i = 0;
    string tmp = "";
    
    //if choice == 0 then generates the same element
    if(choice == 0) {
        while(counter < nB) {
            if(i%26 == 0 && i != 0) {
                file<<"\n";
                counter++;
            }
            file<<"kiwi ";
            counter += 5;
            i++;
        }
    }
    else {  //otherwise generates all different elements (worst case scenario)
        while(counter < nB) {
            tmp = to_string(i)+" ";
            if(i%26 == 0 && i != 0) {
                tmp += "\n";
                counter++;
            }
            file<<tmp;
            counter += tmp.size();
            i++;
        }
    }
    file.close();
    
    cout<<"Number of words: "<<i<<endl;

    return 0;
}