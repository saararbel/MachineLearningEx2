import sys
import numpy as np

def parseDataSet(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter = ',')
    x = data[:, :-1]
    y = data[:, -1]

    return x,y

def calcR(x):
    R = 0
    for idx,xi in enumerate(x):
        lst = xi.tolist()
        if np.math.sqrt(lst[0]*lst[0] + lst[1]*lst[1]) > R:
            R = np.math.sqrt(lst[0]*lst[0] + lst[1]*lst[1])

    return R

def calcGama(x,y):
    gama = sys.float_info.max
    for idx,xi in enumerate(x):
        lst = xi.tolist()
        xi_size = np.math.sqrt(lst[0] * lst[0] + lst[1] * lst[1])
        if xi_size*y[idx] < gama:
            gama = xi_size*y[idx]

    return gama

def cartezian_product(x,w):
    sum = 0
    for i,e in enumerate(x):
        sum += e*w[i]
    return sum

def perceptronAlgo(x, y, learningRate=1):
    weight_vector = [0] * (x[0].size + 1)
    input = []
    bias = 1
    mistakes, iterations = 0, 0
    weights_changed = False
    while not weights_changed:
        weights_changed = True
        for idx,xi in enumerate(x):
            iterations += 1
            xi_list = [bias] + xi.tolist()
            if cartezian_product(xi_list, weight_vector) * y[idx] <= 0:
                weight_vector = [y[idx] * xi_list[w_index] * learningRate + elem for w_index,elem in enumerate(weight_vector)]
                weights_changed = False
                mistakes += 1

    return weight_vector, mistakes, iterations


if __name__ == '__main__':
    x,y = parseDataSet(sys.argv[1])
    w,mistakes, iterations = perceptronAlgo(x,y,0.2)
    print w , mistakes , iterations

    outputFile = open("output.txt", 'w')
    outputFile.write("output1: " + str(w) + "\n")
    outputFile.write("output2: " + str(mistakes)+ "\n")
    outputFile.write("output3: " + str(iterations)+ "\n")

    r = calcR(x)
    gama = calcGama(x,y)
    print r , gama

    print np.math.sqrt((r / gama) * (r / gama))
