#include <iostream>
#include <sstream>
#include <vector>


void printVector(std::vector<int> v) {

  std::stringstream ss;
  for (int x : v) {
    ss << x << " ";
  }
  std::cout << ss.str() << std::endl;
  ss.str(std::string());
}

void insertSort(std::vector<int> &v, bool incremental = true) {

  /*
   *   indexes ->  0  1  2  3  4  5
   *        vector{5, 2, 4, 6, 1, 3}
   *   length = 5
   * Loop from index 1 to the length of the vector
   */
  auto valueFlag = [=](auto vi, auto&& vj) { 
    if (incremental == true) {return vi > vj;}
    return vi < vj; 
  };
  for (int j{1}; j < v.size(); j++) {
    int key = v[j]; // store the value of the index 1
    // Insert v[j] into the sorted sequence v[1..j-1]
    int i = j - 1; // go back 1 in index and store it in i
    /*
     * Loop over the vector from index 0 and j-1
     * while v[i] > v[j]
     *
     * loop invariant has to be true before first iteration
     */

    // bool valueCheck = valueFlag(v[i], key);
    // std::stringstream ss; 
    // ss << "\nIteration = " << i << " | "
    //   << " j = " << j << " | "
    //   << " i = " << i << " | "
    //   << " v[j] = " << v[j]  << " | "
    //   << " v[i] = " << v[i]  << " | "
    //   << " valueCheck = " << valueCheck; 

    // std::cout << ss.str() << std::endl;
    while (i >= 0 && valueFlag(v[i], key)) {
      v[i + 1] = v[i]; // move to the right side
      i -= 1;          // set sentinel i as  previous value of j
    }
    v[i + 1] = key; // move to left side
  }
};



int main(int argc, char *argv[]) {

  std::vector<int> v{5, 2, 4, 6, 1, 3};

  std::cout << "Original Vector -- Size: " << v.size() << std::endl;
  printVector(v);
  // sort vector
  insertSort(v);
  std::cout << "Sorted Vector (incremental) -- Size: " << v.size() << std::endl;
  printVector(v);
  // // sort vector decremental 
  insertSort(v, (bool)false);
  std::cout << "Sorted Vector (decremental) -- Size: " << v.size() << std::endl;
  printVector(v);

  return 0;
}
