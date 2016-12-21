import sys
import numpy as np
import matplotlib.pyplot as plt
import math

def parseDataSet(dataSetFilePath):
    data = np.loadtxt(dataSetFilePath)
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


def perceptronAlgo(x_train, y_train, x_validation, y_validation, learningRate=1):
    weight_vector = [0] * (x_train[0].size + 1)
    number_iterations_acc_not_improved = 40
    mistakes, iterations, bias = 0, 0, 1
    best_validation_accuracy = 0.0
    best_validation_index = -1
    best_weight_vector = weight_vector
    while True:
        iterations += 1
        for idx, xi in enumerate(x_train):
            xi_list = [bias] + xi.tolist()
            if cartezian_product(xi_list, weight_vector) * y_train[idx] <= 0:
                weight_vector = [y_train[idx] * xi_list[w_index] * learningRate + elem for w_index, elem in
                                 enumerate(weight_vector)]
                mistakes += 1

        validation_acc = check_validation_accuracy(x_validation, y_validation,weight_vector)
        if validation_acc > best_validation_accuracy:
            best_validation_index = iterations
            best_validation_accuracy = validation_acc
            best_weight_vector = list(weight_vector)
        if iterations > best_validation_index + number_iterations_acc_not_improved:
            break

        print "Checking validation on Iteration: %s, result: %s" % (str(iterations), str(validation_acc))

    return best_weight_vector, iterations, best_validation_accuracy


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

def plot(w):
    w = np.asarray(w[1:])
    tmp = 1 / (1 + np.exp(-10 * w / max(w)))
    plt.imshow(tmp.reshape(28, 28), cmap="gray")
    plt.draw()
    plt.savefig("final_weight_vector")


if __name__ == '__main__':
    x, y = parseDataSet(sys.argv[1])
    training_size = 0.7
    x_train, y_train, x_validation, y_validation = divide_to_train_and_validation(x, y, training_size)
    w, iterations, best_acc = perceptronAlgo(x_train, y_train, x_validation, y_validation)

    print "Best Accuracy: %s" % str(best_acc)
    print "Final weight vector: %s" % str(w)
    plot(w)

