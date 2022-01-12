"""
Having 2 square matrices, calculate C = AB, let A and B; 

A=  | A00 A01 |     B = | B00 B01 |     C = | C00 C01 |
    | A10 A11 |         | B10 B11 |         | C10 C11 |

C00 = A00 * B00 + A01 * B10 
C01 = A00 * B01 + A01 * B11 
C10 = A10 * B00 + A11 * B10 
C11 = A10 * B01 + A11 * B11 
"""

import numpy as np 
import numpy.typing as npt 

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


def squareMatrixMultiplyRec(A: NDAIntArray | np.int_, 
        B: NDAIntArray | np.int_) -> NDAIntArray | np.int_:
    # For the sake of completion of the chapter, matrices are going to be square 2 x 2 matrices. 
    # TODO: Apply constraints and propeties of matmul in the algorithm. 
    if isinstance(A, np.integer):
        return A*B
    else:
        C00 = squareMatrixMultiplyRec(A[0,0], B[0,0]) 
        + squareMatrixMultiplyRec(A[0,1], B[1,0])

        C01 = squareMatrixMultiplyRec(A[0,0], B[0,1]) 
        + squareMatrixMultiplyRec(A[0,1], B[1,1])

        C10 = squareMatrixMultiplyRec(A[1,0], B[0,0]) 
        + squareMatrixMultiplyRec(A[1,1], B[1,0])

        C11 = squareMatrixMultiplyRec(A[1,0], B[0,1]) 
        + squareMatrixMultiplyRec(A[1,1], B[1,1])
        C = np.vstack((np.hstack((C00, C01)), np.hstack((C10, C11))))
    return C

if __name__ == "__main__":
    A = np.array([[1, 7],
                  [2, 4]])      

    B = np.array([[3, 3],
                  [5, 2]])

    res1 = squareMatrixMultiply(A, B)
    np.testing.assert_array_equal(res1, A @ B)
    print("Using non recursive -> \n", res1)
    res2 = squareMatrixMultiplyRec(A, B)
    np.testing.assert_array_equal(res2, A @ B)
    print("Using non recursive -> \n", res1)

