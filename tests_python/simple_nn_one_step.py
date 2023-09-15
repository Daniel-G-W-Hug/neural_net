import numpy as np


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    # return ((y_true - y_pred) ** 2).mean()
    return (0.5 * (y_pred - y_true) ** 2).mean()


class OurNeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    """

    def __init__(self):
        # Weights
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.w5 = 1
        self.w6 = 1

        # Biases
        self.b1 = 1
        self.b2 = 1
        self.b3 = 1

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.1/4.   # this implementation does now take into account the training size set
        #learn_rate = 0.1   # this implementation does now take into account the training size set
        epochs = 1  # number of times to loop through the entire dataset

        print()
        print("w1= %.5f" % (self.w1))
        print("w2= %.5f" % (self.w2))
        print("b1= %.5f" % (self.b1))
        print()
        print("w3= %.5f" % (self.w3))
        print("w4= %.5f" % (self.w4))
        print("b2= %.5f" % (self.b2))
        print()
        print("w5= %.5f" % (self.w5))
        print("w6= %.5f" % (self.w6))
        print("b3= %.5f" % (self.b3))
        print()

        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Initial loss: %.5f" % (loss))

        for epoch in range(epochs):
            print("epoch: ", epoch)
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                print()
                print("sum_h1= %.5f" % (sum_h1))
                print("h1= %.5f" % (h1))
                print("sum_h2= %.5f" % (sum_h2))
                print("h2= %.5f" % (h2))
                print("sum_o1= %.5f" % (sum_o1))
                print("o1= %.5f" % (o1))
                print()

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                # d_L_d_ypred = -2 * (y_true - y_pred)
                d_L_d_ypred = y_pred - y_true

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                # print()
                # print("d_ypred_d_w5= %.5f" % (d_ypred_d_w5))
                # print("d_ypred_d_w6= %.5f" % (d_ypred_d_w6))
                # print("d_ypred_d_b3= %.5f" % (d_ypred_d_b3))
                # print()

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # print()
                # print("d_h1_d_w1= %.5f" % (d_h1_d_w1))
                # print("d_h1_d_w2= %.5f" % (d_h1_d_w2))
                # print("d_h1_d_b1= %.5f" % (d_h1_d_b1))
                # print()

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # print()
                # print("d_h2_d_w3= %.5f" % (d_h2_d_w3))
                # print("d_h2_d_w4= %.5f" % (d_h2_d_w3))
                # print("d_h2_d_b2= %.5f" % (d_h2_d_b2))
                # print()

                dLdw1 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                dLdw2 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                dLdb1 = d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                dLdw3 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                dLdw4 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                dLdb2 = d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                dLdw5 = d_L_d_ypred * d_ypred_d_w5
                dLdw6 = d_L_d_ypred * d_ypred_d_w6
                dLdb3 = d_L_d_ypred * d_ypred_d_b3

                print("w1= %.5f" % (self.w1))
                print("w2= %.5f" % (self.w2))
                print("w3= %.5f" % (self.w3))
                print("w4= %.5f" % (self.w4))
                print()
                print("w5= %.5f" % (self.w5))
                print("w6= %.5f" % (self.w6))
                print()
                print("b1= %.5f" % (self.b1))
                print("b2= %.5f" % (self.b2))
                print("b3= %.5f" % (self.b3))
                print()

                print("dLdw1= %.5f" % (dLdw1))
                print("dLdw2= %.5f" % (dLdw2))
                print("dLdw3= %.5f" % (dLdw3))
                print("dLdw4= %.5f" % (dLdw4))
                print()
                print("dLdw5= %.5f" % (dLdw5))
                print("dLdw6= %.5f" % (dLdw6))
                print()
                print("dLdb1= %.5f" % (dLdb1))
                print("dLdb2= %.5f" % (dLdb2))
                print("dLdb3= %.5f" % (dLdb3))
                print()

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * dLdw1
                self.w2 -= learn_rate * dLdw2
                self.b1 -= learn_rate * dLdb1

                # Neuron h2
                self.w3 -= learn_rate * dLdw3
                self.w4 -= learn_rate * dLdw4
                self.b2 -= learn_rate * dLdb2

                # Neuron o1
                self.w5 -= learn_rate * dLdw5
                self.w6 -= learn_rate * dLdw6
                self.b3 -= learn_rate * dLdb3

            # --- Calculate total loss at the end of each epoch

            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            print("Epoch %d - loss: %.5f" % (epoch, loss))
            print()
            print("w1= %.5f" % (self.w1))
            print("w2= %.5f" % (self.w2))
            print("w3= %.5f" % (self.w3))
            print("w4= %.5f" % (self.w4))
            print()
            print("w5= %.5f" % (self.w5))
            print("w6= %.5f" % (self.w6))
            print()
            print("b1= %.5f" % (self.b1))
            print("b2= %.5f" % (self.b2))
            print("b3= %.5f" % (self.b3))
            print()


# Define dataset
data = np.array(
    [
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ]
)
all_y_trues = np.array(
    [
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ]
)

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
