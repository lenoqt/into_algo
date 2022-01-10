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

std::tuple<unsigned int, unsigned int, unsigned int>
findMaxCrossingSubArray(std::vector<int> &v, unsigned int low, unsigned int mid,
                        unsigned int high) {

  int leftSum = _INF;
  int rightSum = _INF;
  int sum = 0;
  int maxLeft = 0;
  int maxRight = 0;

  for (int i = mid; i != low; --i) {

    sum = sum + v[i];
    if (sum > leftSum) {

      leftSum = sum;
      maxLeft = i;
    }
  }

  sum = 0;
  for (int j = ++mid; j != high; ++j) {
    sum = sum + v[j];
    if (sum > rightSum) {
      rightSum = sum;
      maxRight = j;
    }
  }

  int totalSum = leftSum + rightSum;
  return std::make_tuple(maxLeft, maxRight, totalSum);
}

std::tuple<unsigned int, unsigned int, unsigned int>
findMaxSubArray(std::vector<int> &v, unsigned int low, unsigned int high) {

  if (high == low) {
    return std::make_tuple(low, high, v[low]);
  }

  else {

    unsigned int mid = (low + high) / 2;

    int leftLow, leftHigh, leftSum;
    std::tie(leftLow, leftHigh, leftSum) = findMaxSubArray(v, low, mid);

    int rightLow, rightHigh, rightSum;
    std::tie(rightLow, rightHigh, rightSum) = findMaxSubArray(v, mid + 1, high);

    int crossLow, crossHigh, crossSum;
    std::tie(crossLow, crossHigh, crossSum) =
        findMaxCrossingSubArray(v, low, mid, high);

    if (leftSum >= rightSum && leftSum >= crossSum) {
      return std::make_tuple(leftLow, leftHigh, leftSum);
    }

    else if (rightSum >= leftSum && rightSum >= crossSum) {
      return std::make_tuple(rightLow, rightHigh, rightSum);
    }

    else {
      return std::make_tuple(crossLow, crossHigh, crossSum);
    }
  }
}
int main(int argc, char *argv[]) {

  std::vector<int> v = {13, -3, -25, 20, -3,  -16, -23, 18,
                        20, -7, 12,  -5, -22, 15,  -4,  7};

  int low{0};
  int mid{7};
  int high{15};
  std::cout << "Using findMaxCrossingSubArray: \n";
  std::tuple<int, int, int> res1 = findMaxCrossingSubArray(v, low, mid, high);
  printTuple(res1);

  std::cout << "Using findMaxSubArray: \n";
  std::tuple<int, int, int> res2 = findMaxSubArray(v, low, high);
  printTuple(res2);

  return 0;
}
