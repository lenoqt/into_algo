#include <climits>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#define _INF INT_MIN

// Helper function to print a tuple
// https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat

template <class Tuple, std::size_t N> struct TuplePrinter {
  static void printTuple(const Tuple &t) {
    TuplePrinter<Tuple, N - 1>::printTuple(t);
    std::cout << ", " << std::get<N - 1>(t);
  }
};

template <class Tuple> struct TuplePrinter<Tuple, 1> {
  static void printTuple(const Tuple &t) { std::cout << std::get<0>(t); }
};

template <class... Args> void printTuple(const std::tuple<Args...> &t) {
  std::cout << "(";
  TuplePrinter<decltype(t), sizeof...(Args)>::printTuple(t);
  std::cout << ")\n";
}

std::tuple<int, int, double> findMaxCrossingSubArray(std::vector<int> &v,
                                                     unsigned int low,
                                                     unsigned int mid,
                                                     unsigned int high) {

  int leftSum = _INF;
  int rightSum = _INF;
  int sum = 0;
  int maxLeft = 0;
  int maxRight = 0;

  for (int i = mid; i != low; i--) {

    sum = sum + v[i];
    if (sum > leftSum) {

      leftSum = sum;
      maxLeft = i;
    }
  }

  for (int j = ++mid; j != high; j++) {
    sum = sum + v[j];
    if (sum > rightSum) {
      rightSum = sum;
      maxRight = j;
    }
  }

  int total = leftSum + rightSum;
  return std::make_tuple(maxLeft, maxRight, total);
}

int main(int argc, char *argv[]) {

  std::vector<int> v = {13, -3, -25, 20, -3,  -16, -23, 18,
                        20, -7, 12,  -5, -22, 15,  -4,  7};

  int low{0};
  int mid{7};
  int high{15};
  std::tuple<int, int, double> res = findMaxCrossingSubArray(v, low, mid, high);
  printTuple(res);

  return 0;
}
