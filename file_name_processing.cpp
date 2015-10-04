#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <iterator>
#include <algorithm>
#include <ctime>

using namespace std;

const string file_name      = "all-samples.dat";
const string folder_name    = "ORL_Face_Database/";

const int nPeople           = 40;
const int nPersonalSamples  = 10;
const int nSamples          = nPeople * nPersonalSamples;

int main(int argc, char *argv[]) {
    ofstream file(file_name.c_str(), ios::out);
    file << nSamples << endl;
    for (int i = 1; i <= nPersonalSamples; ++i) {
        for (int j = 1; j <= nPeople; ++j) {
            file << j << endl;
            file << folder_name << "s" << j << "/" << i << ".pgm" << endl;
        }
    }
    file.close();
}
