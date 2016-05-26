__author__ = 'Shawn'

import sys, csv, copy
import numpy as np
import numpy.matlib as mat
import scipy.spatial.distance as eu

import matplotlib.pyplot as plt

class Cluster:

    def __init__(self,center):
        self.center = center
        self.points = []
        self.labels = []
        self.WCSS = 0

    def setCenter(self,center):
        self.center = center

    def addPoints(self,point,label):
        self.points.append(point)
        self.labels.append(label)

    def resetPoints(self):
        self.points = []
        self.labels = []

    def updateCenter(self):
        self.center = np.array(self.points).mean(0)

    def getCenter(self):
        return  self.center

    def getPoints(self):
        return self.points

    def getLabels(self):
        return self.labels

    def computeWCSS(self):
        sum_list =[]
        for i in self.points:
            dist = eu.euclidean(i,self.center)
            sum_list.append(dist**2)
        self.WCSS = sum(sum_list)
        return self.WCSS



def Kmeans(k,data,labels):
    clusters = {}

    # Create Cluster and Centrids
    for i in range(k):
        clusters[i] = Cluster(data[i])

    # Assign Points to Clusters
    clusterData(k,data,labels,clusters)

    changed = False

    oldcenters = getAllCenters(clusters)

    while True:
        updateCenters(clusters)
        clusterDataAlt(data,labels,clusters)
        centers = getAllCenters(clusters)
        if hasChanged(oldcenters,centers):
            #changed = True
            oldcenters = copy.deepcopy(centers)
        else:
            break

    WCSS = []

    for key,value in clusters.items():
        WCSS.append(value.computeWCSS())

    means, stds, = getMeanandSTD(clusters)

    return WCSS, clusters, means, stds


def getMeanandSTD(clusters):
    means = []
    stds = []
    for k,v in clusters.items():
        mean = np.mean(v.getPoints(),axis=0)
        std = np.std(v.getPoints(),axis=0)
        means.append(mean)
        stds.append(std)
    return means,stds


def hasChanged(old,new):
    oldmean = [np.mean(i) for i in old]
    newmean = [np.mean(i) for i in new]
    cmp = [i for (i, j) in zip(oldmean, newmean) if i != j]
    if cmp:
        return True
    else:
        return False


def clusterData(k,data,labels,clusters):
    for p, l in zip(data[k:], labels[k:]):
        assignCluster(clusters, p, l)


def clusterDataAlt(data,labels,clusters):
    for p, l in zip(data, labels):
        assignCluster(clusters, p, l)


def assignCluster(clusters,point,label):
    dist_list = []
    for key, values in clusters.items():
        dist = eu.euclidean(values.getCenter(),point)
        dist_list.append((dist,key))
    closest = min(dist_list)
    clusters[closest[1]].addPoints(point,label)
    return clusters[closest[1]]


def updateCenters(clusters):
    for key,value in clusters.items():
        value.updateCenter()
        value.resetPoints()

def  getAllCenters(clusters):
    centers = [value.getCenter() for key, value in clusters.items()]
    return centers


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

    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0)
    X_train = (X_train - X_means) / X_stds
    X_test = (X_test - X_means) / X_stds

    #
    X_train = append_ones(X_train)
    X_test = append_ones(X_test)


    WCSSs= []
    RMSEs = []
    ks = [1,2,4,8,16]
    for k in ks:
        WCSS, clusters, means, stds = Kmeans(k, X_train, Y_train)
        WCSSs.append(sum(WCSS))
        np.savetxt("Center, k = " + str(k) + ".csv", getAllCenters(clusters), delimiter=',')

        np.savetxt("Means, k = " + str(k) + ".csv", means, delimiter=',')

        np.savetxt("STDs, k = " + str(k) + ".csv", stds, delimiter=',')

        RMSE = 0


        for x, y in zip(X_test,Y_test):
            cluster = assignCluster(clusters,x,y)
            new_x = x[:-1]
            points = cluster.getPoints()
            points = np.delete(points,-1,1)
            labels = cluster.getLabels()
            unscaleMean = X_means[:len(points)]
            unscaleSTD = X_stds[:len(points)]
            new_x = (new_x * unscaleSTD) + unscaleMean
            points = (points * unscaleSTD) + unscaleMean
            #np.where(~points.any(axis=0))[0]
            beta, _, _, _ = np.linalg.lstsq(points, labels)
            RMSE += (y - np.dot(beta, new_x)) ** 2

        RMSEs.append(np.sqrt((RMSE) / len(X_test)))

    np.savetxt("RSME.csv", RMSEs, delimiter=',')
    np.savetxt("WCSS.cvs", WCSSs,delimiter=',')


    # Plot Graph

    # fig = plt.figure()
    # plt.plot(ks,WCSSs)
    # fig.suptitle("WCSS vs K", fontsize=20)
    # plt.xlabel('K', fontsize=18)
    # plt.ylabel('WCSS', fontsize=16)
    # plt.show()

    # Plot Graph
    fig = plt.figure()
    plt.plot(ks, RMSEs)
    fig.suptitle("RSME vs K", fontsize=20)
    plt.xlabel('K', fontsize=18)
    plt.ylabel('RSME', fontsize=16)
    plt.show()
