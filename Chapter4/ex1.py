import numpy as np 
import numpy.typing as npt 
from typing import Tuple 

NDArrayInt = npt.NDArray[np.int_]

def findMaxCrossing(array: NDArrayInt, 
        low: int, mid: int, high: int) -> Tuple[int, int, float]: 

    #Instantiate variables
    leftSum = float("-inf")
    sum = 0
    maxLeft = maxRight = 0
    # Loop down to low from mid
    for i in range(mid, low, -1):
        sum = sum + array[i]
        if sum > leftSum:
            leftSum = sum 
            maxLeft = i 
    rightSum = float("-inf")
    sum = 0
    for j in range(mid + 1, high):
        sum = sum + array[j]
        if sum > rightSum:
            rightSum = sum 
            maxRight = j 
    return (maxLeft, maxRight, leftSum + rightSum) 


if __name__ == "__main__":
    A = np.array([13, -3, -25, 20, -3, -16,
                  -23, 18, 20, -7, 12, -5,
                  -22, 15, -4, 7 ])
    res = findMaxCrossing(A, 0, 7, 15)
    print(res)
