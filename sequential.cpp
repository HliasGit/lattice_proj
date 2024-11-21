#include "sequential.hpp"

int main(){

    int size = 1024*1024;
    int seed = 100;

    std::vector<int> V(size);
    generate_rand_vec(seed, size, V);

    int min = V[0];


    // Initialize data
    for(int i=0; i<size; i++){
        V[i] = i+4;
    }

    // Set a minimum somewhere
    V[124737] = 1;

    auto start = std::chrono::system_clock::now();
    // Find minimum
    for(int i=0; i<size; i++){
        if(V[i] < min){
            min = V[i];
        }
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Minimum: " << min << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

}

void generate_rand_vec(int seed, int size, std::vector<int> &V){
    srand(seed); 
    generate(V.begin(), V.end(), rand);
}