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


number = input('Insert the sequence of numbers, separated by a space\n')
number_l : List[int] = [int(x) for x in number.split()]

def i_sort(numbs):
    # [5, 2, 4, 6, 1, 3]
    # k=2, i=0 > 
    sorts : List[int] = []
    for j in range(1, len(numbs)):
        key = numbs[j]
        # insert numbs[j] into the sorted sequence numbs[1..j-1]
        i = j - 1
        while i > 0 and sorts[i] > key:
            sorts[i + 1] = sorts[i]
            i =+ 1
        sorts[i + 1] = key
        print(sorts)

if __name__ == '__main__':
    sorta(number_l)
