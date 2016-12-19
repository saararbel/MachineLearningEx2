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

    return x, y


def cartezian_product(x, w):
    sum = 0
    for i, e in enumerate(x):
        sum += e * w[i]
    return sum


def perceptronAlgo(x_tarin, y_train, x_validation, y_validation, learningRate=1):
    weight_vector = [0] * (x_tarin[0].size + 1)
    mistakes, iterations, bias = 0, 0, 1
    check_validation_batch = len(x_tarin) / 12
    best_validation_accuracy = 0.0
    best_validation_improve = True
    best_weight_vector = weight_vector
    while best_validation_improve:
        best_validation_improve = False
        iterations += 1
        for idx, xi in enumerate(x_tarin):
            if idx % check_validation_batch == 0:
                acc = check_validation_accuracy(x_validation, y_validation,weight_vector)
                if acc > best_validation_accuracy:
                    print "Improve !"
                    best_validation_accuracy = acc
                    best_weight_vector = list(weight_vector)
                    best_validation_improve = True
                print "Checking validation on Iteration: %s, idx: %s, result: %s" % (str(iterations), str(idx),
                                                                                     str(acc))
            xi_list = [bias] + xi.tolist()
            if cartezian_product(xi_list, weight_vector) * y_train[idx] <= 0:
                weight_vector = [y_train[idx] * xi_list[w_index] * learningRate + elem for w_index, elem in
                                 enumerate(weight_vector)]
                mistakes += 1
    print "Best acc: %s" % best_validation_accuracy
    print best_weight_vector
    return weight_vector, mistakes, iterations


def check_validation_accuracy(x_validation, y_validation, weight_vector):
    mistakes = 0
    bias = 1
    for idx, xi in enumerate(x_validation):
        xi_list = [bias] + xi.tolist()
        if cartezian_product(xi_list, weight_vector) * y_validation[idx] <= 0:
            mistakes += 1

    return 100 * (1 - (float(mistakes) / len(x_validation)))


def divide_to_train_and_validation(x, y, train_size):
    train_size = (int)(len(x) * train_size)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


if __name__ == '__main__':
    x, y = parseDataSet("")
    x_tarin, y_train, x_validation, y_validation = divide_to_train_and_validation(x, y, 0.8)
    w, mistakes, iterations = perceptronAlgo(x_tarin, y_train, x_validation, y_validation)
    print w, mistakes, iterations
