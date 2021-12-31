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

"""

def merge(array, p, q, r):
    
    n1 = q - p + 1
    n2 = r - q
    l_array = [0 for x in range(n1)]
    r_array = [0 for x in range(n2)]
    for i in range(n1):
        l_array[i] = array[p + i - 1]
    for j in range(n2):
        r_array[j] = array[q + j]
    i = 1
    j = 1 
    for k in range(p, r):
        if l_array[i] <= r_array[j]:
            array[k] = l_array[i]
            i += 1
        else:
            array[k] = r_array[j]
            j += 1

print(merge([2,4,5,7,1,2,3,6,12,11,13,14,15,18,20], 3,4,10))
