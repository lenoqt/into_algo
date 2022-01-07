#include <climits>
#include <iostream>
#include <sstream>
#include <vector>

#define INF INT_MAX
// #define INF ((unsigned)~0)

template <typename T> void printVector(std::vector<T> &v) {

  std::stringstream ss;
  for (int x : v) {
    ss << x << " ";
  }
  std::cout << ss.str() << std::endl;
}

void merge(std::vector<int> &v, const int p, const int q, const int r) {

  int n1 = q - p + 1;
  int n2 = r - q;

  std::vector<int> leftVector(n1);
  std::vector<int> rightVector(n2);

  for (int i = 0; i < n1; i++)
    leftVector[i] = v[p + i];

  for (int j = 0; j < n2; j++)
    rightVector[j] = v[q + 1 + j];

  leftVector.push_back(INF);
  rightVector.push_back(INF);

  int i, j, k;
  i = 0;
  j = 0;
  k = p;

  for (; k <= r; k++) {
    if (leftVector[i] <= rightVector[j]) {
      v[k] = leftVector[i];
      i++;
    } else {
      v[k] = rightVector[j];
      j++;
    }
  }
}

void merge_sort(std::vector<int> &v, const int p, const int r) {
  if (p < r) {

    int q = (p + r) / 2;
    merge_sort(v, p, q);
    merge_sort(v, q + 1, r);
    merge(v, p, q, r);
  }
}

int main(int argc, char *argv[]) {

  int p, r;
  p = 0;
  r = 7;
  std::vector<int> v = {2, 4, 5, 7, 1, 2, 3, 6};
  std::cout << "Vector before sorting\n";
  printVector(v);
  merge_sort(v, p, r);
  std::cout << "Vector after sorting\n";
  printVector(v);

  return 0;
}
