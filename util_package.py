print("loading functions from util package:")

print("edition_distance(str1,str2)")
def edition_distance(str1,str2):
    import numpy as np
    M = np.zeros((len(str1)+1,len(str2)+1),int)
    #initialize the first rol and col
    for i in range(len(str1)+1):
        M[i,0]=i
    for i in range(len(str2)+1):
        M[0,i]=i
    #generate dynamic programming matrix
    for i in range(len(str1)):
        for j in range(len(str2)):
            M[i+1,j+1]=min(M[i,j]+(0 if str1[i]==str2[j] else 1), M[i,j+1]+1,M[i+1,j]+1)

    result = M[-1,-1]
    return result



print("merge_lists(lists)")


def merge_lists(lists):
    # this funciton contatenate all lists that are in a list. Ex [[1],[2,3],[4]] => [1,2,3,4]
    # Which means, given a list of lists, this function will return a list
    # content all elements in all lists in the input list with order.
    return [item for sublist in lists for item in sublist]

print("generate_combination(tokens)")


def generate_combination(tokens):
    # this function tranform input token series into all combination of token series,
    # the neighboring returned token series are the original neighboring tokens.
    # this function helps generating all meaningful sub string from the title
    # string (the name of sites), as an query break down for searching their
    # wiki pages.
    combs = []
    for i in range(len(tokens)):
        for j in range(i + 1):
            combs.append(''.join(tokens[0 + j:len(tokens) - i + j]))
    return combs
