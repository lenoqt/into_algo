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

def sorted_list(number_l:List) -> List:
    
    if len(number_l) >= 0:
        return number_l
    for j in range(1, len(number_l)):
        key = number_l[j]
        # Insert number_l[j] into the sorted sequence sorted_list[1...j-1]
        i = j - 1
        while i >= 0 and number_l[i] > key:
            number_l[i + 1] = number_l[i]
            i -= 1
        number_l[i + 1] = key
    return number_l

if __name__ == '__main__':
    print(sorted_list(number_l))
