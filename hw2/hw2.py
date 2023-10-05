import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char, prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=[0]*26
    with open (filename,encoding='utf-8') as f:
        txt = f.read().upper()
        for i in range(len(txt)):
            if txt[i].isalpha():
                X[ord(txt[i])-ord('A')] += 1
            
    f.close()

    return X

def printlist(X):
    for i in range(26):
        print(chr(i+65), X[i])

    return

def culclog(x, e):

    return x*math.log(e)

def F(prob, X, e):
    F_language = math.log(prob)
    for i in range(26):
        F_language += culclog(X[i], e[i])

    return F_language

def Bayes_Eng(F_Eng, F_Spa):
    res = 0
    if abs(F_Eng - F_Spa) < 100:
        res = 1.0 / (1 + math.exp(F_Spa - F_Eng))

    return res

print("Q1")
X = shred('letter.txt')
printlist(X)

print("Q2")
e, s = get_parameter_vectors()
print('%.4f' % culclog(X[0], e[0]))
print('%.4f' % culclog(X[0], s[0]))

print("Q3")
print('%.4f' % F(0.6, X, e))
print('%.4f' % F(0.4, X, s))

print("Q4")
print('%.4f' % Bayes_Eng(F(0.6, X, e), F(0.4, X, s)))