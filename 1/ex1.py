# Ex 1 of Introduction to algorithms 3rd ed.
# Gustavo Barrios.

"""
The first algorithm and most essential is sorting algorithm, defining a
sequences of numbers {a1, a2, ..., an} of the input sequence such as 
that {a'1 =< a'2 =< ... a'n} for example given: {31, 41, 59, 26, 41, 58},
a sorting algorithm returns as output the sequence {26, 31, 41, 41, 58, 59}.
Such input sequence is called an instance of the sorting problem. In general,
an instance of a problem consist of the input satisfying whatever constraints
are imposed in the problem statement needed to compute the solution.
"""

from typing import List


number = input('Insert the sequence of numbers, separated by a space:\n')
number_l : List[int] = [int(x) for x in number.split()]

def sorted_list(number_l:List) -> List:
    if len(number_l) <= 1:
        return number_l
    for j in range(1, len(number_l)):
        key = number_l[j] # this keeps in memory the value to be moved or not
        # Insert number_l[j] into the sorted sequence sorted_list[1...j-1]
        i = j - 1
        while i >= 0 and number_l[i] > key:
            number_l[i + 1] = number_l[i]
            i -= 1
        number_l[i + 1] = key
    return number_l

if __name__ == '__main__':
    print(sorted_list(number_l))

"""
In this case of insertion sort. Starting with an array of length > than 1,
we indicate j that indicate the current number taken and being inserted into 
the sorted array, the subarray consisting of elements A[1..j-1] constitutes
the current sorted array, and the remaining A[j+1..n] corresponds to the pile 
of number still on the unsorted list, in fact elements A[1..j-1] are the elements
originally in positions 1 through j - 1, but now in sorted order. We state these
properties of A[1..j-1] formally as loop invariant:
    At the start of each iteration for the loop of lines 1-8, the subarray
    A[1..j-1] consists of the elements originally in A[1..j-1], but in sorted order.
    
    The use of loop invariants helps to understand why the algorithm is correct.
    There are 3 things to show on it:
    
    - Initialization: It is true prior to the first iteration of the loop.
    - Maintenance: If it is true before an iteration of the loop, it remains true
    before the next iteration.
    - Termination: When the loop terminates, the invariant gives us a useful 
    property that helps show that the algorithm is correct.

Initialization: Starting that the loop invariant holds before the first loop
    iteration, we check that the list has equal or more than 2 properties.
    When j = 1, the subarray A[1..j-1], consists of just the single element A[0]
    this array, is sorted from the inception, which shows that the invariant
    holds true prior the first iteration.

Maintenance: The second property is to show that each iteration maintain the invariant.
    The for moves the array like A[j-1], A[j-2] and A[j-3] and so on by one 
    position to the right until it finds the proper position for A[j], at which
    it inserts the value of A[j] into the subarray A[1..j] then consists of the
    elements in A[1..j], but in sorted order then increment j for next iteration
    of the for loop preserving the invariant. The while loop runs a second 
    property which has a formal treatment to check whether the value to check
    is greater or equal than the previous in the array.


Termination: We examine what happens when the loop terminates, the block of code
    causing the for loops terminates when when j > A.length = n.
"""
