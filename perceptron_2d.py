import sys

import matplotlib.pyplot as plt
import numpy as np
import math

def import_data(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def calc_r(x):
    R = 0
    for idx, xi in enumerate(x):
        square_size = sum([x*x for x in xi.tolist()])
        if math.sqrt(square_size) > R:
            R = math.sqrt(square_size)

    return R


def calcGama(x, y, normalize_w):
    gama = sys.float_info.max
    for idx, xi in enumerate(x):
        temp = np.dot([1] + xi.tolist(), normalize_w) * y[idx]
        if  temp < gama:
            gama = temp * y[idx]

    return gama


def cartezian_product(x, w):
    sum = 0
    for i, e in enumerate(x):
        sum += e * w[i]
    return sum


def perceptronAlgo(x, y, learningRate=1):
    weight_vector = [0] * (x[0].size + 1)
    input = []
    bias = 1
    mistakes, iterations = 0, 0
    weights_changed = True
    while weights_changed:
        weights_changed = False
        iterations += 1
        for idx, xi in enumerate(x):
            xi_list = [bias] + xi.tolist()
            if np.dot(xi_list, weight_vector) * y[idx] <= 0:
                weight_vector = [(y[idx] * xi_list[w_index] * learningRate) + elem for w_index, elem in
                                 enumerate(weight_vector)]
                weights_changed = True
                mistakes += 1

    return weight_vector, mistakes, iterations


def split_by_sign(y):
    pos_index, neg_index = [], []
    for i, yi in enumerate(y):
        pos_index.append(i) if yi > 0 else neg_index.append(i)

    return pos_index, neg_index


def plot_data(x, y):
    pos_index, neg_index = split_by_sign(y)
    plt.plot([x[i][0] for i in pos_index], [x[i][1] for i in pos_index], 'bo')
    plt.plot([x[i][0] for i in neg_index], [x[i][1] for i in neg_index], 'ro')
    # plt.show()


def vector_normalize(w):
    size = math.sqrt(sum([x*x for x in w]))
    return [float(x/size) for x in w]


def calc_mistake_bound():
    r = calc_r(x)
    gama = calcGama(x, y, vector_normalize(w))
    return (r / gama) * (r / gama)

def plot(w):
    x = np.arange(-30, 30, 0.1)
    final_line = []
    for i in x:
     final_line.append(-i*w[1]/w[2])
    plt.plot(x, final_line)
    plt.show()

if __name__ == '__main__':
    x, y = import_data(sys.argv[1])
    plot_data(x, y)
    w, mistakes, iterations = perceptronAlgo(x, y, 1)
    # plot_separation_hyperplane(w,x)
    plot(w)
    outputFile = open("output.txt", 'w')
    outputFile.write("output1: " + str(w) + "\n")
    outputFile.write("output2: " + str(mistakes) + "\n")
    outputFile.write("output3: " + str(iterations) + "\n")
    outputFile.write("output4: " + str(calc_mistake_bound()) + "\n")

