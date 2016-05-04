import numpy, heapq
import scipy.spatial.distance as eu
from statistics import *


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