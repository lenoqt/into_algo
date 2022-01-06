#include <climits>
#include <iostream>
#include <sstream>
#include <vector>

#define INF INT_MAX
// #define INF ((unsigned)~0)

template <typename T> void printVector(std::vector<T> v) {

  std::stringstream ss;
  for (auto x : v) {
    ss << x << " ";
  }
  std::cout << ss.str() << std::endl;
  ss.str(std::string());
}

template <typename T>
std::vector<T> copyVector(std::vector<T> const &v, const int begin,
                          const int end) {

  std::vector<int>::const_iterator first = v.begin() + begin;
  std::vector<int>::const_iterator last = v.begin() + end + 1;
  std::vector<int> newV(first, last);

  return newV;
}

void merge(std::vector<int> &v, const int p, const int q, const int r) {

  // Initialize left and right vectors with extra slot for sentinel value

  std::vector<int> leftVector;
  std::vector<int> rightVector;
  leftVector = copyVector(v, p, q);
  rightVector = copyVector(v, q + 1, r);
  std::cout << "Left vector\n";
  printVector(leftVector);
  std::cout << "Right vector" << std::endl;
  printVector(rightVector);
  // leftVector.push_back(INF);
  // rightVector.push_back(INF);

  int i, j, k;
  i = 0;
  j = 0;
  k = p;

  for (; k != v.size(); k++) {
    if (leftVector[i] <= rightVector[j]) {
      v[k] = leftVector[i];
      i += 1;
    }
    if (v[k] == rightVector[j]) {
      j += 1;
    }
  }
  leftVector.clear();
  rightVector.clear();
}

void merge_sort(std::vector<int> &v, const int &p, const int &r) {
  if (p < r) {

    int q = (p + r) / 2;
    int comp = p + 1;
    merge_sort(v, p, q);
    merge_sort(v, comp, r);
    merge(v, p, q, r);
  }
}

int main(int argc, char *argv[]) {

  std::vector<int> v{2, 4, 5, 7, 1, 2, 3, 6};
  int p = 0;
  int r = v.size() - 1;
  std::cout << "Vector before sorting\n";
  printVector(v);
  merge_sort(v, p, r);
  std::cout << "Vector after sorting\n";
  printVector(v);

  return 0;
}
