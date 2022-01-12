"""
Having 2 square matrices, calculate C = AB, let A and B; 

A=  | A00 A01 |     B = | B00 B01 |     C = | C00 C01 |
    | A10 A11 |         | B10 B11 |         | C10 C11 |

1) 
C00 = A00 * B00 + A01 * B10 
C01 = A00 * B01 + A01 * B11 
C10 = A10 * B00 + A11 * B10 
C11 = A10 * B01 + A11 * B11 

For any square matrix size let subindices be: 
    A00, A01, A10, A11 = a, b, c, d 
    B00, B01, B10, B11 = e, f, g, h
Solving for 1: 
    C00, C01, C10, C11 = ae + bg, af + bh, ce + dg, cf + dh 

    ae + bg = (a+d)(e+h) + d(g-e) - h(a+b) + (b-d)(g+h)
    af + bh = a(f-h) + h(a+b) 
    ce + dg = e(c+d) + d(g-e)
    cf + dh = a(f-h) + (a+d)(e+h) - e(c+d) - (a-c)(e+f)

    p1 = (a+d)(e+h), p2 = d(g-e)
    p3 = h(a+b), p4 = (b-d)(g+h)
    p5 = a(f-h), p6 = e(c+d)
    p7 = (a-c)(e+f)

    C00 = ae +bg = p1 + p2 - p3 + p4
    C01 = af + bh = p5 + p3 
    C10 = ce + dg = p6 + p2 
    C11 = cf + dh = p5 + p1 - p6 - p7 
"""

import numpy as np 
import numpy.typing as npt 
from typing import Tuple 

NDAIntArray = npt.NDArray[np.int_]

def squareMatrixMultiply(A: NDAIntArray, B: NDAIntArray) -> NDAIntArray:
    # For the sake of completion of the chapter, matrices are going to be square n x n matrices. 
    # TODO: Apply constraints and propeties of matmul in the algorithm. 
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.int_)

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                C[i, j] = C[i, j] + A[i, k]*B[k,j]
    return C

def splitMatrix(M: NDAIntArray) -> Tuple[NDAIntArray, NDAIntArray, NDAIntArray, NDAIntArray]: 
    n = M.shape[0]
    return (M[:n//2, :n//2],
            M[:n//2, n//2:],
            M[n//2:, :n//2],
            M[n//2:, n//2:])


def strassenRecursive(A: NDAIntArray, 
        B: NDAIntArray) -> NDAIntArray:
    n = A.shape[0]
    if n == 1:
        return squareMatrixMultiply(A, B)
    a, b, c, d = splitMatrix(A)
    e, f, g, h = splitMatrix(B)

    p1 = strassenRecursive(a+d, e+h)
    p2 = strassenRecursive(d, g-e)
    p3 = strassenRecursive(a+b, h)
    p4 = strassenRecursive(b-d, g+h)
    p5 = strassenRecursive(a, f-h)
    p6 = strassenRecursive(c+d, e)
    p7 =  strassenRecursive(a-c, e+f)

    C00 = p1 + p2 - p3 + p4
    C01 = p5 + p3 
    C10 = p6 + p2 
    C11 = p5 + p1 - p6 - p7 

    return np.vstack((np.hstack((C00, C01)), np.hstack((C10, C11))))

if __name__ == "__main__":
    A = np.array([[1, 7],
                  [2, 4]])      

    B = np.array([[3, 3],
                  [5, 2]])

    res1 = squareMatrixMultiply(A, B)
    np.testing.assert_array_equal(res1, A @ B)
    print("Using non recursive -> \n", res1)
    res2 = strassenRecursive(A, B)
    np.testing.assert_array_equal(res2, A @ B)
    print("Using recursive -> \n", res1)

