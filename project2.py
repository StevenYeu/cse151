import numpy, math, csv, heapq
import scipy.spatial.distance as eu
import numpy.matlib as mat
from statistics import *
import sys
#Shiyao Liu, A10626758
#Steven Yeu, A10720198

class sample:
    def __init__(self, filename):
        self.sampleCounter = {}
        self.filename = filename
        with open(self.filename) as f:
            self.sampleSize = sum(1 for _ in f)
        self.Nn = 0
        self.Nr = 0
        numpy.random.seed(1)
        self.genRandom = 0
        self.matrix = []
        self.training_set = []
        self.test_set = []
        self.training_label = []
        self.test_label = []

    def setCounter(self):
        for i in range(self.sampleSize):
            self.sampleCounter[i] = 0

    def setPickNext(self):
        self.Nr = self.sampleSize
        self.Nn = round(self.sampleSize*0.1)

    def sampler(self, iters):
        for _ in range(iters):
            self.setPickNext()
            for i in range(self.sampleSize):
                self.genRandom = numpy.random.rand()

                if self.Nr == 0:
                    break
                pickP = self.Nn/self.Nr
                if self.genRandom < pickP:
                    self.Nn -= 1
                    self.sampleCounter[i] += 1
                self.Nr -= 1

        mean = numpy.mean(list(self.sampleCounter.values()))/float(iters)
        std = numpy.std(list(self.sampleCounter.values()))/float(iters)

        return mean, std

    def load(self):
        if self.filename == 'abalone.data':

            # create proxy variables for gender
            gender = []
            with open(self.filename, 'r') as f:
                for line in f:
                    gender.append(line[0])

            male = [1 if i == 'M' else 0 for i in gender]
            female = [1 if i == 'F' else 0 for i in gender]
            infant = [1 if i =='I' else 0 for i in gender]
            self.matrix = numpy.array(list( numpy.loadtxt(self.filename, delimiter=",",usecols=(1,2,3,4,5,6,7,8))))
            self.matrix = numpy.insert(self.matrix,0,male,axis=1)
            self.matrix = numpy.insert(self.matrix,0,female,axis=1)
            self.matrix = numpy.insert(self.matrix,0,infant,axis=1)
        else:
            self.matrix = numpy.array(list(csv.reader(open(self.filename,"r"),delimiter=','))).astype('float')

    def split(self):

        # split into test and training

        # create index to test and train data
        training_index = [k for k,v in self.sampleCounter.items() if v == 0]
        test_index = [k for k,v in self.sampleCounter.items() if v == 1]

        # create training data
        self.training_set = [self.matrix[i] for i in training_index]
        _, training_size = numpy.shape(self.training_set)
        self.training_set = numpy.array(self.training_set)
        self.training_label = self.training_set[:, training_size-1]
        self.training_set = self.training_set[:, 0:training_size-1]

        # create test data
        self.test_set = [self.matrix[i] for i in test_index]
        _, test_size = numpy.shape(self.test_set)
        self.test_set = numpy.array(self.test_set)
        self.test_label = self.test_set[:, test_size-1]
        self.test_set = self.test_set[:, 0:test_size-1]

    def zscale(self, setname):
        # calculate mean and std
        row, size = numpy.shape(setname)
        setname = numpy.array(setname)
        mean_list = [numpy.mean(setname[:, i]) for i in range(size)]
        std_list = [numpy.std(setname[:, i]) for i in range(size)]
        mean_matrix = mat.repmat(mean_list, row, 1)
        std_matrix = mat.repmat(std_list, row, 1)

        # z scale
        setname = (setname - mean_matrix)/std_matrix
        return setname

    def set_training(self, alist):
        self.training_set = alist

    def set_test(self, alist):
        self.test_set = alist

    def kNN(self, k):
        error = 0
        prediction = []
        result = []
        _, size = numpy.shape(self.training_set)
        for ind, i in enumerate(self.test_set):
            temp = [(eu.euclidean(i, data), self.training_label[index]) for index, data in enumerate(self.training_set)]
            k_smallest = [item[1] for item in heapq.nsmallest(k, temp)]
            try:
                majority_label = mode(k_smallest)
            # Error check in case  there is no mode
            except StatisticsError:
                majority_label = max(set(k_smallest), key=k_smallest.count)

            if self.test_label[ind] != majority_label:
                error += 1
            prediction.append(majority_label)
        error_rate = error/float(len(self.test_set))*100.0
        return error_rate, prediction, self.test_label


def confusionMatrix(actual,pred):
    con = numpy.zeros((max(actual)+1,max(actual)+1))

    z = zip(actual,pred)
    for (a,b) in z:
        con[int(a)][int(b)] += 1

    # write confusion matrix to file
    numpy.savetxt("matrix.csv",con,delimiter=',')

    #output = open('output.txt', "w")
    #for item in con:
    #    output.write("%s\n" % item)
    return con

if __name__ == '__main__':
    sample = sample(sys.argv[1])
    sample.setCounter()
    sample.sampler(1)
    sample.load()
    sample.split()
    sample.set_training(sample.zscale(sample.training_set))
    sample.set_test(sample.zscale(sample.test_set))
    res_dict = {}
    for i in range(1, 10, 2):
        rate, prediction, result = sample.kNN(i)
        res_dict[rate] = (prediction, result,i)
    #print(i, rate)

    prediction, result,k = res_dict[min(res_dict)]
    print(min(res_dict),k)
    print(confusionMatrix(result, prediction))







