import numpy, heapq
import scipy.spatial.distance as eu
from statistics import *
import functools as fun


def confusionMatrix(actual,pred):
    con = numpy.zeros((max(actual)+1,max(actual)+1))

    z = zip(actual,pred)
    for (a,b) in z:
        con[int(a)][int(b)] += 1

    # write confusion matrix to file
    numpy.savetxt("matrix.csv",con,delimiter=',')
    return con


def kNN(k, training_set, test_set,training_label, test_label):
    error = 0
    prediction = []
    result = []
    _, size = numpy.shape(training_set)
    for ind, i in enumerate(test_set):
        temp = [(eu.euclidean(i, data), training_label[index]) for index, data in enumerate(training_set)]
        k_smallest = [item[1] for item in heapq.nsmallest(k, temp)]
        try:
            majority_label = mode(k_smallest)
        # Error check in case  there is no mode
        except StatisticsError:
            majority_label = max(set(k_smallest), key=k_smallest.count)

        if test_label[ind] != majority_label:
            error += 1
        prediction.append(majority_label)
    error_rate = error/float(len(test_set))*100.0
    return error_rate, prediction, test_label


def back_solve(A, b):
    m, n = A.shape
    x = b
    newN = n -1
    for i in range(n):
        if i > 0:
            for j in range(0, i):
                x[newN - i] = x[newN - i] - x[newN - j] * A[newN - i][newN - j]
        x[newN - i] = x[newN - i] / (1.0 * A[newN - i][newN - i])
    return x

def qr_decompose(X):
    n, d = numpy.shape(X)
    Qlist = []
    R = X
    #Qacc =numpy.identity(n)
    for i in range(d):
        # 1. obtains the target column zi
        z = R[i:, i:i+1]
        # 2. find vi
        e = numpy.zeros((n-i, 1))
        e[0][0] = 1
        if z[0][0] > 0:
            v = -numpy.linalg.norm(z)*e - z
        else:
            v = numpy.linalg.norm(z)*e - z
        # 3. find Householder matrix Pi
        P = numpy.identity(n-i) - numpy.dot(2*v, v.T)/numpy.dot(v.T,v)
        # 4. Q
        Q = numpy.identity(n)
        Q[i:,i:] = P
        Qlist.append(Q)
        #Qacc = numpy.dot(Q,Qacc)
        # 5. Update R
        R = numpy.dot(Q,R)
    Qacc = fun.reduce(lambda x, y: numpy.dot(x,y), Qlist[::-1])
    return Qacc.T, R

if __name__ == '__main__':
    A = [[1, -1, -1], [1, 2, 3], [2, 1, 1], [2, -2, 1], [3, 2, 1]]
    A = numpy.array(A)
    Q, R = qr_decompose(A)
    print(Q)
    print(R)

    G = numpy.array([[1,-2,1],[0,1,6],[0,0,1]])
    b = numpy.array([4,-1,2])
    print(back_solve(G,b))
