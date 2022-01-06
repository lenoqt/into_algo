# type: ignore
"""
The ex1.py is a insertion sort algorithm which takes time roughly equal to c1n^2
to sort n items, where c1 is a constant that does not depend on n. That is, it 
takes time roughly proportional to n^2.

The following code implements the divide-and-conquer approach which many useful
algorithms are recursive in structure: to solve a given problem, they call 
themselves recursively one or more times to deal with closely related sub-problems.
These algorithms typically follow a divide-and-conquer approach: they break the 
problem into several subproblems that are similar to the original problem 
but smaller in size, solve the subproblems recursively, and then combine these
solutions to create a solution to the original problem.

The divide-and-conquer paradigm involves three steps at each level of the recursion:

- Divide the problem into a number of subproblems that are smaller instances of 
the same problem.

- Conquer the subproblems by solving them recursively. If the subproblem sizes 
are small enough, however, just solve the subproblems in a straighforward manner.

- Combine the solutions to the subproblems into the solution for the original 
problem.

The merge sort algorithm closely follows the divide-and-conquer paradigm, 
intuitively, it operates as follows.

- Divide: Divide each n-element sequence to be sorted into two subsequences of 
n/2 elements each.

- Conquer: Sort the two subsequences recursively using merge sort.

- Combine: Merge the two sorted subsequences to produce the sorted answer.

The recursion "bottoms out" when the sequence to be sorted has lenght 1, in which
case there is no work to be done, since every sequence of lenght 1 is already in
sorted order.

Having an array A with indices p, q and r such p <= q < r 
- Divide into A[p..q] (left array) and A[q + 1...r] 
- Do it recursively until having a sorted small array 
- Merge then back 
ie. 
                 0  1  2  3  4  5  6  7
            A   [2, 4, 5, 7, 1, 2, 3, 6]
                /                    \
        L [2, 4, 5, 7]          R [1, 2, 3, 6]
          /         \              /         \
    L [2, 4]    R [5, 7]      L [1, 2]    R [3, 6]
       /   \       /   \         /   \      /    \  
   L [2] R [4] L [5] R [7]   L [1] R [2] L [3] R [6]
      \    /      \    /        \   /       \   /  
    L [2, 4]    R [5, 7]      L [1, 2]    R [3, 6]
         \         /              \          / 
       L [2, 4, 5, 7]           R [1, 2, 3, 6]
                \                   /  
            A*  [1, 2, 2, 3, 4, 5, 6]

"""
from copy import deepcopy 

def merge_sort(array):
    """
    The implementation has to be sightly different as python does in-place replacement of indexes 
    """
    if len(array) > 1:
    # For a array lenght 8, p = 0, q = 3, r = 7
    # Where: p is the start index of the subarray, q middle index of the subarray 
    # and r the end of subarray.
    # for python indexes have to be adjusted as they start from 0
        r = len(array)//2 
        # split the original array into Left and Right arrays 
        left_array = array[:r]
        right_array = array[r:]
        print("L -> ", left_array, "R -> ", right_array)
        merge_sort(left_array)
        merge_sort(right_array)
        # Loop over left array 
        left_array.append(float('inf')) 
        right_array.append(float('inf'))

        i = j = 0 

        for k in range(0, len(array)):
            if left_array[i] < right_array[j]:
                array[k] = left_array[i]
                i += 1 
            else:
                array[k] = right_array[j]
                j += 1



if __name__ == "__main__":
    A = [2, 4, 5, 7, 1, 2, 3, 6]
    B = deepcopy(A)
    merge_sort(B)
    print("Using merge sort --> ", A, "-->", B)
    print("Using built-in sort --> ", A, "-->", sorted(A))

    C = [122, 42321, 52333, 0, 31, 42, 4, 89, 12981, 298382, 828382]
    D = deepcopy(C)
    merge_sort(D)
    print("Using merge sort --> ", C, "-->", D)
    print("Using built-in sort --> ", C, "-->", sorted(C))

