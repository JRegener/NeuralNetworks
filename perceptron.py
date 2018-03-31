import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmX):
    return (1 - sigmX) * sigmX


# def add_bias(x, size):
#     bias = np.full((size, 1), 1)
#     return np.concatenate((x, bias), axis=1)
#
# class NeuralNetwork:
#     def __init__(self, layers):
#         self.weights = []
#         # hidden weights
#         for i in range(1, len(layers) - 1):
#             r = 2 * np.random.random((layers[i - 1] + 1, layers[i])) - 1
#             self.weights.append(r)
#         # output weights
#         r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
#         self.weights.append(r)
#
#         # print(self.weights)
#
#     def train(self, x, y, epochs=10000):
#         # for e in range(epochs):
#         l0 = add_bias(x, len(x))
#         l1 = add_bias(sigmoid(np.dot(l0, self.weights[0])), len(l0))
#         l2 = sigmoid(np.dot(l1, self.weights[1]))
#         error = ((y - l2) ** 2) / 1
#
#         # out: (ideal - actual)*derivative(sigmoid)
#         dout = (y - l2) * sigmoid_derivative(l2)
#         # print(dout)
#
#         dhide = []
#         for i in range(len(l1)):
#             # print(dout[i])
#             # print(self.weights[1])
#             # print(dout)
#             # print((dout[i] * self.weights[1]).T)
#             r = l1[i] * (dout[i] * self.weights[1]).T
#             dhide.append(r)
#
#
#
#         # print(dhide)
#
#         # plt.plot(l2, error, 'bo', l2, error, 'k')
#         # plt.xlabel(r'$weight$')
#         # plt.ylabel(r'$err$')
#         # plt.grid(True)
#         # plt.show()


if __name__ == "__main__":
    # nn = NeuralNetwork([2, 2, 1])
    # layers: input 2 hidden 2 output 1 [2, 2, 1]
    # train set
    I = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    O = np.array([[1, 0, 0, 1]])

    weight1 = 2 * np.random.random((2, 2)) - 1
    weight2 = 2 * np.random.random((1, 2)) - 1

    bias1 = 2 * np.random.random((2, 4)) - 1
    bias2 = 2 * np.random.random((1, 4)) - 1

    alpha = 0.2
    for i in range(1000000):
        l0 = I.T
        l1 = sigmoid(np.dot(weight1, l0)+bias1)
        l2 = sigmoid(np.dot(weight2, l1)+bias2)

        # if i % 10000 == 0:
        #     print("iteration: ", i)
        #     print(l2)

        error = O - l2
        if i % 10000 == 0:
            print("iteration: ", i)
            print(np.mean(np.abs(error)))
            print(l2)

        dOut = error.T * sigmoid_derivative(l2).T
        dHide = sigmoid_derivative(l1).T * np.dot(dOut, weight2)
        dInput = sigmoid_derivative(l0).T * np.dot(dHide, weight1)

        weight2 += alpha * np.dot(l1, dOut).T
        weight1 += alpha * np.dot(l0, dHide).T
