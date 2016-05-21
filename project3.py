__author__ = 'Shawn'

import sys, csv, Functions
import numpy as np


if __name__ == '__main__':
    file = sys.argv[1]
    if file == 'abalone.data':
        # create proxy variables for gender
        gender = []
        with open(file, 'r') as f:
            for line in f:
                gender.append(line[0])
            male = [1 if i == 'M' else 0 for i in gender]
            female = [1 if i == 'F' else 0 for i in gender]
            infant = [1 if i =='I' else 0 for i in gender]
            data = np.array(list( np.loadtxt(file, delimiter=",",usecols=(1,2,3,4,5,6,7,8))))
            data = np.insert(data,0,male,axis=1)
            data = np.insert(data,0,female,axis=1)
            data = np.insert(data,0,infant,axis=1)
    else:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = np.array([[float(r) for r in row] for row in reader])



    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.6)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]


    Q, R = Functions.qr_decompose(X_train)
    m = R.shape[1]
    Rhat = R[:m][:m+1]
    chat =  np.dot(Q.T, Y_train)[:m]
    beta = Functions.back_solve(Rhat, chat)

    print(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))






