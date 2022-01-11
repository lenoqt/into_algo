import numpy as np 
import numpy.typing as npt 

NDAIntArray = npt.NDArray[np.int_]

def squareMatrixMultiply(A:NDAIntArray, B:NDAIntArray) -> NDAIntArray:
    # For the sake of completion of the chapter, matrices are going to be square n x n matrices. 
    # TODO: Apply constraints and propeties of matmul in the algorithm. 

    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.int_)

    for i in range(0, n):
        for j in range(0, n):
            C[i, j] = 0
            for k in range(0, n):
                C[i, j] = C[i, j] + A[i, k]*B[k,j]
    return C

if __name__ == "__main__":
    A = np.array([[1, 7],
                  [2, 4]])      

    B = np.array([[3, 3],
                  [5, 2]])

    res = squareMatrixMultiply(A, B)
    np.testing.assert_array_equal(res, A @ B)
    print(res)

