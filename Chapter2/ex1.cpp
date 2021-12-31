#include <iostream>
#include <sstream>
#include <vector>

void insert_sort(std::vector<int> &v) {

  for (int j{1}; j < v.size(); j++) {
    int key = v[j];
    // Insert v[j] into the sorted sequence v[1..j-1]
    int i = j - 1;
    while (i >= 0 && v[i] > key) {
      v[i + 1] = v[i];
      i -= 1;
    }
    v[i + 1] = key;
  }
};

int main(int argc, char *argv[]) {

  std::stringstream ss;

  std::vector<int> v{5, 2, 4, 6, 1, 3};

  std::cout << "Original Vector" << std::endl;
  for (int x : v) {
    ss << x << " ";
  }
  std::cout << ss.str() << std::endl;
  ss.str(std::string());
  insert_sort(v);

  std::cout << "Sorted Vector" << std::endl;
  for (int x : v) {
    ss << x << " ";
  }
  std::cout << ss.str() << std::endl;
  return 0;
}
