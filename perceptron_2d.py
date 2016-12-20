import sys

import matplotlib.pyplot as plt
import numpy as np


def import_data(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def calc_r(x):
    R = 0
    for idx, xi in enumerate(x):
        lst = xi.tolist()
        if np.math.sqrt(lst[0] * lst[0] + lst[1] * lst[1]) > R:
            R = np.math.sqrt(lst[0] * lst[0] + lst[1] * lst[1])

    return R


def calcGama(x, y):
    gama = sys.float_info.max
    for idx, xi in enumerate(x):
        lst = xi.tolist()
        xi_size = np.math.sqrt(lst[0] * lst[0] + lst[1] * lst[1])
        if xi_size * y[idx] < gama:
            gama = xi_size * y[idx]

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
        for idx, xi in enumerate(x):
            iterations += 1
            xi_list = [bias] + xi.tolist()
            if cartezian_product(xi_list, weight_vector) * y[idx] <= 0:
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
    plt.show()


# def plot_separation_hyperplane(w, original_x):
#     """
#     This is an attempt to solve 1.(f)
#     :rtype: object
#     """
#     hyperplane = '(%s)*x+1*(%s)' % (w[2] / w[1], w[0] / w[1])
#         min_x = min(original_x[::1])
#     x = np.array(range(min_x, 10))
#     evaled = eval(hyperplane)
#     plt.plot(x, evaled)
#     plt.show()


if __name__ == '__main__':
    x, y = import_data(sys.argv[1])
    # plot_data(x, y)
    w, mistakes, iterations = perceptronAlgo(x, y, 0.70)
    # plot_separation_hyperplane(w,x)
    print w, mistakes, iterations


    outputFile = open("output.txt", 'w')
    outputFile.write("output1: " + str(w) + "\n")
    outputFile.write("output2: " + str(mistakes) + "\n")
    outputFile.write("output3: " + str(iterations) + "\n")

    # r = calcR(x)
    # gama = calcGama(x, y)
    # print r, gama

    # print (r / gama) * (r / gama)
