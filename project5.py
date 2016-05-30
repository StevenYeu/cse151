import numpy as np
import sys, csv, math
from functools import reduce
if __name__ == '__main__':
    file = sys.argv[1]

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array( [ [float(r) for r in row] for row in reader])
        print(data.shape)
        data = data[:-1]
    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.9)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]

    sumofCol = np.sum(X_train, axis = 0)
    cnt = 0
    spam = [i for i in Y_train if i == 1]
    spam = len(spam)
    ham = len(Y_train) - spam
    Pspam = float(spam)/len(Y_train)
    Pham = float(ham)/len(Y_train)

    WordGivenSpamList = []
    WordGivenHamList = []
    for i in X_train:
        X_data = zip(i, Y_train)
        cntWordGivenSpam = reduce(lambda acc, elm: acc + elm[0] if elm[1] else acc + 0, X_data,0)
        cntWorldGivenHam = reduce(lambda acc, elm: acc + elm[0] if not elm[1] else acc + 0, X_data, 0)
        WordGivenSpamList.append(cntWordGivenSpam)
        WordGivenHamList.append(cntWorldGivenHam)
    sumWGSL = sum(WordGivenSpamList)
    sumWGHL = sum(WordGivenHamList)
    numofTerms = len(WordGivenHamList)
    PwordGivenSpamLst = [float(i+1)/(sumWGSL + numofTerms) for i in WordGivenSpamList]
    PwordGivenHamLst = [float(i+1)/(sumWGHL + numofTerms) for i in WordGivenHamList]

