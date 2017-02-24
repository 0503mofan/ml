import numpy
import operator


#def CreatDataSet():
   #group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.1], [0.0, 0.0]])
    #labels = ['A', 'A', 'B', 'B']
    #return group, labels

group = numpy.array([[1.0, 1.1],
               [1.0, 1.0],
               [0.0, 0.1],
               [0.0, 0.0]])
labels = ['A', 'A', 'B', 'B']

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True
                              )
    return  sortedClassCount[0][0]

print(classify0([0, 0], group, labels, 3))
