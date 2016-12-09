import numpy as np
import sys


def parseDataSet(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter = ',')
    x = data[:, :-1]
    y = data[:, -1]

    return x,y


def calcR(x):
    R = 0
    for idx,xi in enumerate(x):
        lst = xi.tolist()
        if(lst[0]*lst[0] + lst[1]*lst[1] > R):
            R = lst[0]*lst[0] + lst[1]*lst[1]

    return R

def calcGama(x,y):
    gama = sys.float_info.max
    for idx,xi in enumerate(x):
        lst = xi.tolist()
        if(y[idx]*lst[0] + y[idx]*lst[1] < gama):
            gama = y[idx]*lst[0] + y[idx]*lst[1]

    return gama


def perceptronAlgo(x, y, learningRate=1):
    w = [0] * 3
    input = []
    mistakes, iterations = 0,0
    weightsChanged = False
    while not weightsChanged :
        weightsChanged = True
        for idx,xi in enumerate(x):
            iterations += 1
            lst = xi.tolist()
            x1,x2 = lst[0] , lst[1]
            if ((w[0] * 1 + w[1] * x1 + w[2] * x2) * y[idx]) <= 0:
                w = [y[idx]*1*learningRate + w[0], y[idx]*x1*learningRate + w[1],y[idx]*x2*learningRate + w[2]]
                weightsChanged = False
                mistakes += 1

    return w, mistakes, iterations


if __name__ == '__main__':
    x,y = parseDataSet(sys.argv[1])
    w,mistakes, iterations = perceptronAlgo(x,y,0.3)
    print w , mistakes , iterations

    outputFile = open("output.txt", 'w')
    outputFile.write("output1: " + str(w) + "\n")
    outputFile.write("output2: " + str(mistakes)+ "\n")
    outputFile.write("output3: " + str(iterations)+ "\n")

    r = calcR(x)
    gama = calcGama(x,y)

    # print (r/gama)*(r/gama)




