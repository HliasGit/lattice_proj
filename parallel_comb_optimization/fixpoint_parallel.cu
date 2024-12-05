#include <iostream>
#include <vector>
#include <chrono>
#include <stack>
#include <sstream>
#include <cuda_runtime.h>

#include "parser.hpp"

struct Node {
  int depth; // depth in the tree
    std::vector<int> value; // configuration for that node
    bool* domain;
    int* lenDom;
    int domSize;

    // Constructor
    Node(size_t N, int* lenDom, int domSize)
        : depth(0), value(N), lenDom(lenDom), domSize(domSize) {
        domain = new bool[domSize];
        for (int i = 0; i < domSize; i++) {
            domain[i] = true; // Initialize domain to true
        }
    }

    // Copy Constructor (Deep Copy)
    Node(const Node& other)
        : depth(other.depth), value(other.value), lenDom(other.lenDom), domSize(other.domSize) {
        domain = new bool[domSize];
        std::copy(other.domain, other.domain + domSize, domain);
    }

    // Move Constructor
    Node(Node&& other) noexcept
        : depth(other.depth), value(std::move(other.value)), domain(other.domain), lenDom(other.lenDom), domSize(other.domSize) {
        other.domain = nullptr;
    }

    // Destructor
    ~Node() {
        delete[] domain;
    }

    // Assignment Operator (Deep Copy)
    Node& operator=(const Node& other) {
        if (this != &other) {
            depth = other.depth;
            value = other.value;
            lenDom = other.lenDom;
            domSize = other.domSize;

            delete[] domain;
            domain = new bool[domSize];
            std::copy(other.domain, other.domain + domSize, domain);
        }
        return *this;
    }

    // Move Assignment Operator
    Node& operator=(Node&& other) noexcept {
        if (this != &other) {
            depth = other.depth;
            value = std::move(other.value);
            lenDom = other.lenDom;
            domSize = other.domSize;

            delete[] domain;
            domain = other.domain;
            other.domain = nullptr;
        }
        return *this;
    }

  public:

  // Right methods
  void printLenDom(){
    for(int i = 0; i < value.size(); i++){
      std::cout << lenDom[i] << " ";
    }
    std::cout << std::endl;
  }

  bool getAvail(int index)
  {
    return domain[index];
  }

  void turnToFalse(int index)
  {
    domain[index] = false;
  }

  int take_value_of_singleton(int var){
    for(int i=lenDom[var-1]; i<lenDom[var]; i++){
      if(domain[i]){
        // std::cout << "Var: " << var << ", position: " << i << " and is " << domain[i] << std::endl;
        return i;
      }
    }
    return -1;
  }

  int takeFirstNotSingDomain(){
    for(depth=0; depth<value.size(); depth++){
      int count = 0;
      for(int dom_n=lenDom[depth-1]; dom_n<lenDom[depth]; dom_n++){
        if(domain[dom_n]) count++;
        if (count == 2) return depth;
      }
    }
    return -1;
  }

  bool checkPossibleInstantiation(int abs_index){
    return domain[abs_index];
  }

  void enforceSingleton(int var, int value){
    for(int i=lenDom[var-1]; i<lenDom[var]; i++){
      if (i != value) {
        domain[i] = false;
      }
    }
  }
  
  void printValue(){
    for(int i = 0; i < value.size(); i++){
      std::cout << value[i] << " ";
    }
    std::cout << std::endl;
  }

  void printAvail(){
    for(int i = 0; i < domSize; i++){
      std::cout << domain[i] << " ";
    }
    std::cout << std::endl;
  }

  int count_singleton(){
    int singleton = 0;
    for(int var=0; var<value.size(); var++){
      int count = 0;
      for(int dom_n=lenDom[var-1]; dom_n<lenDom[var]; dom_n++){
        if(domain[dom_n]) count++;
      }
      if(count == 1) singleton++;
    }
    return singleton;
  }

};

// Function to generate arrays based on upper bounds
int generateArrays(int* ub, size_t size, int *lenDom) {
    std::vector<std::vector<int>> arrays(size);
    for (size_t i = 0; i < size; ++i) {
        arrays[i].resize(ub[i] + 1);
        for (int j = 0; j <= ub[i]; ++j) {
            arrays[i][j] = j;
        }
    }
    lenDom[0] = arrays[0].size();
    int sum = arrays[0].size();

    for(int i = 1; i < size; i++){
        lenDom[i] = lenDom[i-1] + arrays[i].size();
        sum += arrays[i].size();
    }
    return sum;
}

bool isSafe(const std::vector<int>& values, const int j, int **C, const int depth)
{
  for (int i = 0; i < depth; i++) {
    if (C[i][depth] == 1 && values[i] == j) {
      return false;
    }
  }
  return true;
}

__device__ int getDomain(int* lenghtsOfDom, int value){
  if(value == -1) return 0;
  else return lenghtsOfDom[value];
}

__global__ void parallelSetting(int var, int value_of_singleton_rel, int* lenghtsOfDom, bool *domain, int* C, int N){
  int other_var = threadIdx.x;

  if(var == other_var){
  
  } else {
    // printf("SONO QUA 2");
    if(value_of_singleton_rel<(getDomain(lenghtsOfDom, (other_var))-getDomain(lenghtsOfDom, (other_var-1))) && C[var*N+other_var] == 1){ // If the constraint is satisfied and the value of the singleton is less than the length of the domain of the variable compared
      // printf("SONO QUA 3");
      domain[getDomain(lenghtsOfDom, (other_var-1))+value_of_singleton_rel] = false; // Remove the value from the domain of the other variable
    }
  }
}


void fixpoint_iter(Node &node, std::stack<Node>& pool, int N, size_t &num_sol, int **C, int* C_lin, int* d_C) {

  int numberOfSingleton = node.count_singleton(); // Get the number of singletons so far

  bool singleton_gen = true;
  while(singleton_gen){ // Enter the loop
    singleton_gen = false;

    for(int var=0; var<N; var++){ // Loop over the variables
      int count = 0;

      for(int dom_numb=node.lenDom[var-1]; dom_numb<node.lenDom[var]; dom_numb++){
        if(node.getAvail(dom_numb)){ // Check how many values of the domain are available
          count++;
        }
      }

      if (count == 1){ // If only one than the domain is singleton
        int value_of_singleton_rel = node.take_value_of_singleton(var) - node.lenDom[var-1]; // Gives the postion of the singleton, relative to the domain of its variable.
        if(value_of_singleton_rel == -1) {
          std::cout << "Problem with finding a singleton " << std::endl;
          return;
        }

        int *d_length_of_dom;
        bool *d_domain;

        // Allocate memory for the device
        cudaMalloc(&d_length_of_dom, N*sizeof(int));
        cudaMalloc(&d_domain, node.domSize*sizeof(bool));
        

        cudaMemcpy(d_length_of_dom, node.lenDom, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_domain, node.domain, node.domSize*sizeof(bool), cudaMemcpyHostToDevice);

        // Here I need to give to every thread (except when var == other_var) one variable
        parallelSetting<<<1,N>>>(var, value_of_singleton_rel, d_length_of_dom, d_domain, d_C, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        cudaMemcpy(node.domain, d_domain, node.domSize*sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_length_of_dom);
        cudaFree(d_domain);

        int newNumbSing = node.count_singleton(); // Count again the number of singletons



        if(newNumbSing > numberOfSingleton){ //If it changed than the fixpoint made another singleton so we can remove other values from the domains
          singleton_gen = true;
          numberOfSingleton = newNumbSing;
        }
        
      }
      else if (count == 0) return;
      else {
        // std::cout << "No singleton for variable " << var << std::endl;
      }
    }
  
  }

  if(node.count_singleton() == N){ // A solution is found when all the variables have singleton domains
    num_sol++;
    // every 100000 solutions print the solution
    if(num_sol % 100000 == 0){
      std::cout << "Solution found: " << num_sol << std::endl;
    }
    return;
  }

  int first_not_sing_domain = node.takeFirstNotSingDomain(); // Take the first variable that has no singleton domain 
  if(first_not_sing_domain == -1){
    std::cout << "Problem with the domain " << std::endl;
  }  

  // Branch and bound the latter
  for(int index=node.lenDom[first_not_sing_domain-1]; index<node.lenDom[first_not_sing_domain]; index++){
    // std::cout << "Branching " << index << ", for variable: " << first_not_sing_domain << std::endl;
    if(node.checkPossibleInstantiation(index)){
      Node child(node);
      child.value[first_not_sing_domain] = index;
      child.enforceSingleton(first_not_sing_domain, index);
      child.depth++;
      pool.push(std::move(child));
    }
  }

}

int main(int argc, char** argv) {

    // Check if the number of arguments is correct
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " elements " << std::endl;
        exit(1);
    }

    // number of elements to treat
    size_t N = std::stoll(argv[1]);
    std::cout << "Solving " << N << " sized generic problem\n" << std::endl;

    // use data generated by the parser
    int domainSize = 0;
    int lenDom[N];

    // Constraint matrix
    int **C;

    // Read the input file
    Data data;
    std::ostringstream filename;
    filename << "pco_" << N << ".txt";

    if (data.read_input(filename.str().c_str())) {
        std::cout << "Number of elements: " << data.get_n() << std::endl;
        N = data.get_n();

        int *ub = data.get_u(); // Upper bounds
        domainSize = generateArrays(ub, N, lenDom);
        

        std::cout << "Domain size: " << domainSize << std::endl << std::endl;

        // get the constraint matrix
        C = data.get_C();

    } else {
        std::cerr << "Error while reading the file" << std::endl;
        return 1;
    }

    int *C_lin = new int[N*N];

    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){
        C_lin[i*N+j] = C[i][j];
      }
    }

    int *d_C;

    cudaMalloc(&d_C, N*N*sizeof(int));
    cudaMemcpy(d_C, C_lin, N*N*sizeof(int), cudaMemcpyHostToDevice);

    // Initialization of the root node
    Node root(N, lenDom, domainSize);

    // Initialization of the pool of nodes
    std::stack<Node> pool;
    pool.push(std::move(root));

    // Statistics to check correctness
    size_t exploredSol = 0;

    // Beginning of the Depth-First tree-Search
    auto start = std::chrono::steady_clock::now();

    // Start the fix-point iteration
    while (pool.size() != 0) {
        Node currentNode(std::move(pool.top())); // Get the top of the stack
        pool.pop(); // Remove the top of the stack

        fixpoint_iter(currentNode, pool, N, exploredSol, C, C_lin, d_C); // Fix-point iteration
    }

    cudaFree(d_C);

    auto end = std::chrono::steady_clock::now(); // End of the search
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // Duration of the search

    // outputs
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "Total solutions: " << exploredSol << std::endl;

    return 0;
}