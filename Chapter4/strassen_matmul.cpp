#include <iostream>
#include <sstream>
#include <vector>

typedef std::vector<std::vector<int>> matrix;
void printMatrix(matrix &m) {

  std::stringstream ss;
  for (std::vector<int> x : m) {
    for (int y : x) {
      ss << y << " ";
    }
    ss << "\n";
  }
  std::cout << ss.str() << std::endl;
}

matrix squareMatrixMul(matrix &A, matrix &B) {

  unsigned int n = A.size();
  std::vector<std::vector<int>> C(n, std::vector<int>(n, 0));
  for (int i = 0; i != n; i++) {

    for (int j = 0; j != n; j++) {

      for (int k = 0; k != n; k++) {

        C[i][j] = C[i][j] + A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

int main(int argc, char *argv[]) {

  matrix A = {{1, 7}, {2, 4}};

  matrix B = {{3, 3}, {5, 2}};

  matrix C = squareMatrixMul(A, B);
  printMatrix(A);
  std::cout << "*\n\n"; 
  printMatrix(B);
  std::cout << " =\n\n";
  printMatrix(C);
  return 0;
}
