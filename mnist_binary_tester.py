import sys
import numpy as np

def parseDataSet(dataSetFilePath):
    data = np.loadtxt("D:\\Excresies\\MachineLearning\\ex2\\dataset\\training_data_1_vs_8.rs.dat.gz")
    x = data[:, 1:]
    y = data[:, 0]
    palette = [1, 8]
    key = np.array([1, -1])
    index = np.digitize(y.ravel(), palette, right=True)
    y = key[index].reshape(y.shape)

    return x,y

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
    x, y = parseDataSet(sys.argv[1])
    w, mistakes, iterations = perceptronAlgo(x, y, 0.3)
    print w, mistakes, iterations

