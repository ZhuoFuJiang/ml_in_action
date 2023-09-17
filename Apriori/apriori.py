import numpy as np


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    扫面频繁项
    :param D: 数据集
    :param Ck: 候选集列表
    :param minSupport:
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.get(can, None):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = np.float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLK = len(Lk)
    # 这里的k-2还用到了一些数学技巧
    # [0, 1], [0, 2], [1, 2] 只比较第一个元素，且只对第一个元素相等的集合求取并操作会更快
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # 从至少拥有两个元素的集合中创建规则
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    # m为单元素长度
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


def getActionIds():
    actionIdList = []
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = 1
        except Exception as e:
            print(e)


if __name__ == "__main__":
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print(C1)
    D = list(map(set, dataSet))
    L1, supportData0 = scanD(D, C1, 0.5)
    print(L1)
    L, supportData1 = apriori(dataSet, minSupport=0.5)
    print(L)
    rules = generateRules(L, supportData1, minConf=0.3)
    print(rules)

    # 毒蘑菇的相似特征
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, supportData = apriori(mushDataSet, minSupport=0.3)
    for item in L[1]:
        if item.intersection('2'):
            print(item)
