import numpy as np

dic = {}
def dictionary(corpus):#Paperwork place
    listtop10 = []
    listall = []
    for sentence in corpus:
        for word in sentence:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
                listall.append(word)
    a = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    num=0
    for word in a:
     listtop10.append(word[0])
     num=num+1
     if(num==10):
         break
    return listtop10, listall

def centerpoint(list, word2idx=None, wvecs=None):
    x=0
    listindex = []
    for word in list :
        if word in word2idx:
         listindex.append(word2idx[word])
    for position in listindex:
        x=x+wvecs[position]
    result=x/len(listindex)
    return result
def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

def allvector(list, word2idx=None):
    listindex = {}
    for word in list:
        if word in word2idx:
         listindex[word]=word2idx[word]
    return list
def compare(dict,verctor):
    list=[]
    num=1
    replace_list=[]
    for word in dict:
        if EuclideanDistances(dict[word],verctor)<=num:
            list.append(word)
    return list