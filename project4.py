__author__ = 'Shawn'

import sys, csv, Functions, copy
import numpy as np
import functools as fun
import numpy.matlib as mat
import scipy.spatial.distance as eu


def zscale(setname):
    # calculate mean and std
    row, size = np.shape(setname)
    setname = np.array(setname)
    mean_list = [np.mean(setname[:, i]) for i in range(size)]
    std_list = [np.std(setname[:, i]) for i in range(size)]
    mean_matrix = mat.repmat(mean_list, row, 1)
    std_matrix = mat.repmat(std_list, row, 1)

    # z scale
    setname = (setname - mean_matrix)/std_matrix
    return setname

def cluster(k, data):
    np.random.seed(0)
    np.random.shuffle(data)
    cluster_list = [[] for i in range(int(k))]
    center_list = [data[i] for i in range(int(k))]
    new_center_list = []
    center_means = [np.mean(i) for i in center_list]
    cluster_set(k, data, center_list, cluster_list)
    Changed = False
    while not Changed:
        update_centers(cluster_list, new_center_list)
        cluster_set(k, data, new_center_list, cluster_list)
        #print(center_list)
        #print(new_center_list)
        print(np.array(center_list).shape)
        print(np.array(new_center_list).shape)
        oldmean = [np.mean(i) for i in center_list]
        newmean = [np.mean(i) for i in new_center_list]
        cmp_l = [i for (i,j) in zip(oldmean,newmean) if i != j]
        if cmp_l:
            Changed = True
            center_list = copy.deepcopy(new_center_list)
        else:
            break

    sum_list = []

    for n, i in list(enumerate(center_list)):
        ed_list = []
        for j in cluster_list[n]:
            d = eu.euclidean(i, j)
            ed_list.append(d**2)
        sum = fun.reduce(lambda x, y: x + y, ed_list)
        sum_list.append(sum)

    mean = [np.array(i).mean(0) for i in cluster_list]
    std = [np.array(i).std(0) for i in cluster_list]
    return center_list, sum_list, mean, std

def cluster_set(k, data, center_list, cluster_list):
    for j in data[k:]:
        d_list = []
        for i in center_list:
            d_list.append(eu.euclidean(i, j))
        m = min(d_list)
        ind = d_list.index(m)
        cluster_list[ind].append(j)



def update_centers(cluster_list, new_center_list):
    for i in cluster_list:
        centroid = np.array(i).mean(0)
        new_center_list.append(centroid)

if __name__ == '__main__':
    file = sys.argv[1]

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

    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.9)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]

    X_train = zscale(X_train)

    center, WCSS, mean, std = cluster(int(sys.argv[2]), X_train)
    print(center)
    print(WCSS)
    print(mean)
    print(std)
    #beta,_,_,_  = np.linalg.lstsq(X_train,Y_train)

    #print(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))