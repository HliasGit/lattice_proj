#include <iostream>
#include <vector>
#include <chrono>
#include <stack>
#include <sstream>
#include <cuda_runtime.h>

#include "parser.hpp"


__global__ void parallel(bool* domain, int* lenDom, int *singletonFlags, int **C, int N, int domSize, int *numberOfSingleton, int var){

    int other_dom = threadIdx.x; 
    other_dom = (other_dom < N) ? other_dom : 0;

    int count = 0;

    // std::cout << "Variable: " << var << std::endl;
    // std::cout << "Going from " << node.lenDom[var-1] << " to " << node.lenDom[var] << std::endl;
    for(int dom_numb=lenDom[var-1]; dom_numb<lenDom[var]; dom_numb++){
      if(domain[dom_numb]){
        count++;
      }
    }


    if (count == 1){

      int value_of_singleton_abs = 0;
      for(int i=lenDom[var-1]; i<lenDom[var]; i++){
        if(domain[i]){
          // std::cout << "Var: " << var << ", position: " << i << " and is " << domain[i] << std::endl;
          value_of_singleton_abs = i;
          break;
        }
      }

      int value_of_singleton_rel = value_of_singleton_abs - lenDom[var-1]; // Gives the postion of the singleton, relative to the domain of its variable.
      //std::cout << "Value of singleton: " << value_of_singleton_rel << " for variable " << var << std::endl;
      if(value_of_singleton_rel == -1) {
        //std::cout << "Problem with finding a singleton " << std::endl;
        return;
      }


      for(int other_dom=0; other_dom<N; other_dom++){
        //std::cout << "other_dom " << other_dom << std::endl; 

        if(C[var][other_dom] == 1 && value_of_singleton_rel<(lenDom[other_dom]-lenDom[other_dom-1])){
          domain[lenDom[other_dom-1]+value_of_singleton_rel] = false;

          int singleton = 0;
          for(int var=0; var<N; var++){
            int count = 0;
            for(int dom_n=lenDom[var-1]; dom_n<lenDom[var]; dom_n++){
              if(domain[dom_n]) count++;
            }
            if(count == 1) singleton++;
          }
          //std::cout << "New number of singletons: " << newNumbSing << std::endl;
          if(singleton > numberOfSingleton[0]){
            singletonFlags[0] = true;
            numberOfSingleton[0] = singleton;
          }
        }
      }
    }
    else if (count == 0) return;
    else {
      // std::cout << "No singleton for variable " << var << std::endl;
    }
  
}

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

void generateDomain(int* ub, size_t size, bool* domain, int domainSize) {
    for (size_t i = 0; i < domainSize;++i) {
      domain[i] = true;
    }
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


void fixpoint_iter(Node &node, std::stack<Node>& pool, int N, size_t &num_sol, int **C){

  int numberOfSingleton = node.count_singleton();
  // std::cout << "Number of singletons: " << numberOfSingleton << std::endl;
  int count = 0;

  bool singleton_gen = true;
  while(singleton_gen){
    singleton_gen = false;

    std::cout << "The available values are: " << std::endl;
    node.printAvail();
    std::cout << std::endl;


    for(int var=0; var<N; var++ ){
      int* d_singletonFlags;
      int* d_nSing;

      cudaMallocManaged(&node.domain, node.domSize * sizeof(bool));
      cudaMallocManaged(&node.lenDom, N * sizeof(int));
      cudaMallocManaged(&d_singletonFlags, sizeof(int));
      cudaMallocManaged(&C, N * N * sizeof(int*));
      cudaMallocManaged(&d_nSing, 1 * sizeof(int*));

      parallel<<<1,N>>>(node.domain, node.lenDom, d_singletonFlags, C, N, node.domSize, d_nSing, var);
      cudaDeviceSynchronize();

      singleton_gen = d_singletonFlags[0];
    }

    
    if (count == 2) exit(1);
    count++;
  }

  if(node.count_singleton() == N){
    num_sol++;
    return;
  }

  int first_not_sing_domain = node.takeFirstNotSingDomain(); 
  if(first_not_sing_domain == -1){
    std::cout << "Problem with the domain " << std::endl;
  }
  // std::cout << "First not singleton domain: " << first_not_sing_domain << std::endl;
  

  // Branch
  // std::cout << "Branching" << std::endl << std::endl;
  for(int index=node.lenDom[first_not_sing_domain-1]; index<node.lenDom[first_not_sing_domain]; index++){
    // print the avail values of node
    // std::cout << "The available values for the iter " << index << " are: " << std::endl;
    // node.printAvail();
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
    bool* domain;
    int domainSize = 0;
    int lenDom[N];

    int **C;

    Data data;
    std::ostringstream filename;
    filename << "pco_" << N << ".txt";

    // Use the dynamically constructed filename in the data.read_input function call
    if (data.read_input(filename.str().c_str())) {
        // print the number of elements
        std::cout << "Number of elements: " << data.get_n() << std::endl;
        N = data.get_n();

        // get the upper bound
        int *ub = data.get_u();
        domainSize = generateArrays(ub, N, lenDom);
        domainSize++;

        std::cout << "Domain size: " << domainSize << std::endl << std::endl;

        domain = new bool[domainSize];

        generateDomain(ub, N, domain, domainSize);

        // get the constraint matrix
        C = data.get_C();

    } else {
        std::cerr << "Error while reading the file" << std::endl;
        return 1;
    }


    // initialization of the root node (the board configuration where no queen is placed)
    Node root(N, lenDom, domainSize);

    // initialization of the pool of nodes (stack -> DFS exploration order)
    std::stack<Node> pool;
    pool.push(std::move(root));

    // statistics to check correctness (number of nodes explored and number of solutions found)
    size_t exploredSol = 0;

    // beginning of the Depth-First tree-Search
    auto start = std::chrono::steady_clock::now();

    int count = 0;

    while (pool.size() != 0) {
        // get a node from the pool
        Node currentNode(std::move(pool.top()));
        pool.pop();

        fixpoint_iter(currentNode, pool, N, exploredSol, C);
        count ++;
        if (count==1) exit(1);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    //std::cout << "Count " << count << std::endl;

    // outputs
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "Total solutions: " << exploredSol << std::endl;

    return 0;
}