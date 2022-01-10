import numpy as np 
import numpy.typing as npt 
from typing import Tuple 

NDArrayInt = npt.NDArray[np.int_]


# TODO: Fix numpy copying data, do it with view()

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


def findMaxSubarray(array: NDArrayInt, low: int, high: int) -> Tuple[int, int, float]:

    # Base case for 1 element
    if high == low: 
        return (low, high, array[low])
    else:
        mid = (low + high) // 2
        
        leftLow, leftHigh, leftSum = findMaxSubarray(array, low, mid)
        rightLow, rightHigh, rightSum = findMaxSubarray(array, mid + 1, high)

        crossLow, crossHigh, crossSum = findMaxCrossing(array, low, mid, high) 

        if leftSum >= rightSum and leftSum >= crossSum: 
            return (leftLow, leftHigh, leftSum)
        
        elif rightSum >= leftSum and rightSum >= crossSum: 
            return (rightLow, rightHigh, rightSum) 

        else:
            return (crossLow, crossHigh, crossSum)


if __name__ == "__main__":
    A = np.array([13, -3, -25, 20, -3, -16,
                  -23, 18, 20, -7, 12, -5,
                  -22, 15, -4, 7 ])
    res1 = findMaxCrossing(A, 0, 7, 15)
    res2 = findMaxSubarray(A, 0, 15)
    assert res1 == res2

    print("Result of findMaxCrossing: ", res1)
    print("Result of findMaxSubarray: ", res2)

