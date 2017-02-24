import numpy
import operator


group = numpy.array([[1.0, 1.1],
               [1.0, 1.0],
               [0.0, 0.1],
               [0.0, 0.0]])
labels = ['A', 'A', 'B', 'B']

diffMat = numpy.tile([0,0], (group.shape[0], 1)) - group
sqDiffMat = diffMat**2
sqDistance = sqDiffMat.sum(axis=1)
distances = sqDistance**0.5
sortedDistIndicies = distances.argsort()
classCount = {}
for i in range(3):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
r = operator.itemgetter(1)
print(r)
type(r)
