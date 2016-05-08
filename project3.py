__author__ = 'Shawn'

import sys, csv, Functions
import numpy as np


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array([[float(r) for r in row] for row in reader])

    # create random sample train (80%) and test set (20%)
    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.8)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]

    print("Xtrain", X_train.shape)
    print("Xtest", X_test.shape)

    print("Ytrain", Y_train.shape)
    print("Ytest", Y_test.shape)

    Q, R = Functions.qr_decompose(X_train)
    print("Done With QR")
    beta = Functions.back_solve(R, np.dot(Q.T, Y_train))

    print(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))





