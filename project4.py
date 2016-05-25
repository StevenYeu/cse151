__author__ = 'Shawn'

import sys, csv, Functions, copy
import numpy as np
import functools as fun
import numpy.matlib as mat
import scipy.spatial.distance as eu
import Functions
from sklearn.metrics import mean_squared_error


def append_ones(m):
    return np.concatenate((m, np.ones((m.shape[0], 1))), axis=1)


def zscale(setname):
    # calculate mean and std
    row, size = np.shape(setname)
    setname = np.array(setname)
    mean_list = [np.mean(setname[:, i]) for i in range(size)]
    std_list = [np.std(setname[:, i]) for i in range(size)]
    mean_matrix = mat.repmat(mean_list, row, 1)
    std_matrix = mat.repmat(std_list, row, 1)

    # z scale
    newmat = (setname - mean_matrix)/std_matrix
    return newmat

def cluster(k, data, labels):
    cluster_list = [[] for i in range(int(k))]
    center_list = [data[i] for i in range(int(k))]
    cluster_list,cluster_label = cluster_set(k, data, center_list, cluster_list, labels)
    Changed = False
    while not Changed:
        old_center_list = copy.deepcopy(center_list)
        center_list = update_centers(cluster_list)
        cluster_list, cluster_label = cluster_set(k, data, center_list, cluster_list,labels)
        oldmean = [np.mean(i) for i in old_center_list]
        newmean = [np.mean(i) for i in center_list]
        cmp_l = [i for (i,j) in zip(oldmean,newmean) if i != j]
        if cmp_l:
            Changed = True
        else:
            break

    sum_list = []

    for n, i in list(enumerate(center_list)):
        ed_list = []
        for j in cluster_list[n]:
            d = eu.euclidean(i, j)
            ed_list.append(d**2)
        sum = fun.reduce(lambda x, y: x + y, ed_list,0)
        sum_list.append(sum)

    mean = [np.array(i).mean(0) for i in cluster_list]
    std = [np.array(i).std(0) for i in cluster_list]
    print(np.array(cluster_list).shape[0])
    return cluster_list, center_list, sum_list, mean, std, cluster_label

def cluster_set(k, data, center_list, cluster_list,cluster_label):
    new_labels = [[]  for i in range(len(cluster_list))]
    new_cluster = [[]  for i in range(len(cluster_list))]
    for n,j in list(enumerate(data[k:])):
        d_list = []
        b = n+k
        for i in center_list:
            d_list.append(eu.euclidean(i, j))
        m = min(d_list)
        ind = d_list.index(m)
        new_cluster[ind].append(j)
        new_labels[ind].append(cluster_label[b])
    return new_cluster,new_labels



def update_centers(cluster_list):
    new_list = [[] for i in range(len(cluster_list))]
    j = 0
    for i in cluster_list:
        centroid = np.array(i).mean(0)
        new_list[j] = (centroid)
        j = j+1
    return  new_list


def getCluster(cluster_list, center_list, point,labels):
    dist = [eu.euclidean(i,point) for i in center_list]
    closest = min(dist)
    index = dist.index(closest)
    return cluster_list[index],labels[index]



if __name__ == '__main__':
    file = sys.argv[1]

    # create proxy variables for gender
    with open('abalone.data', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            s = row[0]
            sex = [1, 0, 0] if s == 'F' else ([0, 1, 0] if s == 'M' else [0, 0, 1])
            data.append(sex + [float(r) for r in row[1:]])
        data = np.array(data)

    np.random.seed(0)
    np.random.shuffle(data)
    train_num = int(data.shape[0] * 0.9)
    X_train = data[:train_num, :-1]
    Y_train = data[:train_num, -1]
    X_test = data[train_num:, :-1]
    Y_test = data[train_num:, -1]

    #X_train = zscale(X_train)
    # X_means = np.mean(X_train, axis=0)
    # X_stds = np.std(X_train, axis=0)
    # X_train = (X_train - X_means) / X_stds
    # X_test = (X_test - X_means) / X_stds

    X_train = append_ones(X_train)
    X_test = append_ones(X_test)


    clusters, center, WCSS, mean, std, lables = cluster(int(sys.argv[2]), X_train, Y_train)
    # print("Center")
    # print(center)
    # print("WCSS")
    # print(sum(WCSS))
    # print("Mean")
    # print(mean)
    # print("STD")
    # print(std)

    RSME = 0
    for c,l in zip(clusters,lables):
        #model,lab = getCluster(clusters, center, i, lables)
        #model = np.array(model)
        beta, _, _, _ = np.linalg.lstsq(c,l)
        print(mean_squared_error(np.dot(X_test,beta),Y_test))
        print(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))

    #beta,_,_,_  = np.linalg.lstsq(X_train,Y_train)
    print(RSME)
    #print(np.sqrt(np.mean((np.dot(X_test, beta) - Y_test) ** 2)))