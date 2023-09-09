import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    生成随机质心
    :param dataSet: 数据集
    :param k: 质心个数
    :return:
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            # .A表示将矩阵转换为数组格式
            ptsInclust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInclust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分K-均值算法，为了解决局部最小值问题，SSE分为两簇后portlandClubs.txt最大程度降低
    :param dataSet: 数据集
    :param k: 质心数目
    :param distMeas: 距离计算函数
    :return:
    """
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            # 计算其他簇的SSE
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # replace a centroid with two best centroids
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # reassign new clusters, and SSE
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment


def distSLC(vecA, vecB):
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * np.cos(
        np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split("\t")
        datList.append([np.float(lineArr[4]), np.float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p',
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsIncurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsIncurrCluster[:, 0].flatten().A[0],
                    ptsIncurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0],
                myCentroids[:, 1].flatten().A[0],
                marker='+', s=300)
    plt.show()


if __name__ == "__main__":
    dataMat = np.mat(loadDataSet('testSet2.txt'))
    print(randCent(dataMat, 2))
    print(distEclud(dataMat[0], dataMat[1]))
    myCentroids, clustAssing = biKmeans(dataMat, 3)
    # 绘制散点图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].A, dataMat[:, 1].A)
    ax.scatter(myCentroids[:, 0].A, myCentroids[:, 1].A, color='r')
    plt.show()

    clusterClubs()



