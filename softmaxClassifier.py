"""
File: softmaxClassifier.py
Language: Python 3.5.1
Author: Karan Jariwala( kkj1811@rit.edu )
Description: Implemented a softmax classifier using using stochastic gradient descent
with mini-batches and momentum to minimize softmax (cross-entropy) loss with L2
weight decay regularization of this single layer neural network.
"""

__author__ = "Karan Jariwala"

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Train and Test filenames
TRAIN_FILENAME = "iris-train.txt"
TEST_FILENAME = "iris-test.txt"

class Softmax:
    """
    A softmax classifier
    """
    __slots__ = ("epochs", "learningRate", "batchSize", "regStrength", "wt", "momentum", "velocity")
    def __init__(self, epochs, learningRate, batchSize, regStrength, momentum):
        """
        Softmax constructor which initialized parameters
        :param epochs: Number of iterations over complete training data
        :param learningRate: A step size or a learning rate
        :param batchSize: A mini-batch size(less than total number of training data)
        :param regStrength: A regularization strength
        :param momentum: A momentum value
        """
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.regStrength = regStrength
        self.momentum = momentum
        self.velocity = None
        self.wt = None

    def train(self, xTrain, yTrain, xTest, yTest):
        """
        Train a softmax classifier model on training data using stochastic gradient descent with mini-batches
        and momentum to minimize softmax (cross-entropy) loss of this single layer neural network. It calcualtes
        mean per-class accuracy for the training/testing data and the loss.
        :param xTrain: Training input data
        :param yTrain: Training labels
        :param xTest: Testing input data
        :param yTest: Testing labels
        :return: A tuple of training/Testing losses and Accuracy
        """
        D = xTrain.shape[1]  # dimensionality
        label = np.unique(yTrain)
        numOfClasses = len(label) # number of classes
        yTrainEnc = self.oneHotEncoding(yTrain, numOfClasses)
        yTestEnc = self.oneHotEncoding(yTest, numOfClasses)
        self.wt = 0.001 * np.random.rand(D, numOfClasses)
        self.velocity = np.zeros(self.wt.shape)
        trainLosses = []
        testLosses = []
        trainAcc = []
        testAcc = []
        for e in range(self.epochs): # loop over epochs
            trainLoss = self.SGDWithMomentum(xTrain, yTrainEnc)
            testLoss, dw = self.computeLoss(xTest, yTestEnc)
            trainAcc.append(self.meanAccuracy(xTrain, yTrain))
            testAcc.append(self.meanAccuracy(xTest, yTest))
            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainAcc : {:.7f}\t|\tTestAcc: {:.7f}"
                  .format(e, trainLoss, testLoss, trainAcc[-1], testAcc[-1]))
        return trainLosses, testLosses, trainAcc, testAcc

    def SGDWithMomentum(self, x, y):
        """
        Stochastic gradient descent with mini-batches. It divides training data into mini-batches
        and compute loss and grad on that mini-batches and updates the weights. It repeats for all samples.
        :param x: An input samples
        :param y: An input labels
        :return: Total loss computed
        """
        losses = []
        # Randomly juggle up the data.
        randomIndices = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randomIndices]
        y = y[randomIndices]
        for i in range(0, x.shape[0], self.batchSize):
            Xbatch = x[i:i+self.batchSize]
            ybatch = y[i:i+self.batchSize]
            loss, dw = self.computeLoss(Xbatch, ybatch)
            self.velocity = (self.momentum * self.velocity) + (self.learningRate * dw)
            self.wt -= self.velocity
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def softmaxEquation(self, scores):
        """
        It calculates a softmax probability
        :param scores: A matrix(wt * input sample)
        :return: softmax probability
        """
        scores -= np.max(scores)
        prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
        return prob

    def computeLoss(self, x, yMatrix):
        """
        It calculates a cross-entropy loss with regularization loss and gradient to update the weights.
        :param x: An input sample
        :param yMatrix: Label as one-hot encoding
        :return:
        """
        numOfSamples = x.shape[0]
        scores = np.dot(x, self.wt)
        prob = self.softmaxEquation(scores)

        loss = -np.log(np.max(prob)) * yMatrix
        regLoss = (1/2)*self.regStrength*np.sum(self.wt*self.wt)
        totalLoss = (np.sum(loss) / numOfSamples) + regLoss
        grad = ((-1 / numOfSamples) * np.dot(x.T, (yMatrix - prob))) + (self.regStrength * self.wt)
        return totalLoss, grad

    def meanAccuracy(self, x, y):
        """
        It calculates mean-per class accuracy
        :param x: Input sample
        :param y: label sample
        :return: mean-per class accuracy
        """
        predY = self.predict(x)
        predY = predY.reshape((-1, 1))  # convert to column vector
        return np.mean(np.equal(y, predY))

    def predict(self, x):
        """
        It predict the label based on input sample and a model
        :param x: Input sample
        :return: predicted label
        """
        return np.argmax(x.dot(self.wt), 1)

    def oneHotEncoding(self, y, numOfClasses):
        """
        Convert a vector into one-hot encoding matrix where that particular column value is 1 and rest 0 for that row.
        :param y: Label vector
        :param numOfClasses: Number of unique labels
        :return: one-hot encoding matrix
        """
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not numOfClasses:
            numOfClasses = np.max(y) + 1
        yMatrix = np.zeros((len(y), numOfClasses))
        yMatrix[np.arange(len(y)), y] = 1
        return yMatrix


def plotGraph(trainLosses, testLosses, trainAcc, testAcc):
    """
    Plot a Epochs vs. Cross Entropy Loss graph
    :param trainLosses: List of training loss over every epochs
    :param testLosses: List of testing loss over every epochs
    :param trainAcc: List of training accuracy over every epochs
    :param testAcc: List of testing accuracy over every epochs
    :return: None
    """
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Epochs vs. Cross Entropy Loss")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainAcc, label="Train Accuracy")
    plt.plot(testAcc, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Epochs vs. Mean per class Accuracy")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()

def readData(filename):
    """
    Read data from file and divide into input sample and a label.
    :param filename: name of a file
    :return: input sample and label
    """
    dataMatrix = np.loadtxt(filename)
    np.random.shuffle(dataMatrix)
    X = dataMatrix[:, 1:]
    y = dataMatrix[:, 0].astype(int)
    y = y.reshape((-1, 1))
    y -= 1
    return X, y

def makeMeshGrid(x, y, h=0.02):
    """
    Create a mesh point to plot decision boundary.
    :param x: data or sample (for x-axis on meshgrid)
    :param y: label(for y-axis on meshgrid)
    :param h: step size for meshgrid
    :return: matrix of x-axis and y-axis
    """
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plotContours(plt, model, xx, yy, **params):
    """
    It plot a contour.
    :param plt: A matplotlib.pyplot object
    :param model: softmax classifier model
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param params: Number of parameters to pass to contour function
    :return:
    """
    arr = np.array([xx.ravel(), yy.ravel()])
    scores = np.dot(arr.T, sm.wt)
    prob = model.softmaxEquation(scores)
    Z = np.argmax(prob, axis=1) + 1

    # Put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)
    # plt.axis('off')

def plotDecisionBoundary(x, y):
    """
    Plot a decision boundary to display a sample with region
    :param x: input data or sample
    :param y: label
    :return: None
    """
    markers = ('+', '.', 'x')
    colors = ('blue', 'dimgrey', 'maroon')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    xx, yy = makeMeshGrid(x, y)
    plotContours(plt, sm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    for idx, cl in enumerate(np.unique(y)):
        xBasedOnLabel = x[np.where(y[:,0] == cl)]
        plt.scatter(x=xBasedOnLabel[:, 0], y=xBasedOnLabel[:, 1], c=cmap(idx),
                    cmap=plt.cm.coolwarm, marker=markers[idx], label=cl)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Feature X1")
    plt.ylabel("Feature X2")
    plt.title("Softmax Classifier on Iris Dataset(Decision Boundary)")
    plt.xticks()
    plt.yticks()
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    """
    Main method
    """
    trainX, trainY = readData(TRAIN_FILENAME) # Training data
    testX, testY = readData(TEST_FILENAME) # Testing data

    sm = Softmax(epochs=1000, learningRate=0.07, batchSize=10, regStrength=0.001, momentum=0.05)
    trainLosses, testLosses, trainAcc, testAcc = sm.train(trainX, trainY, testX, testY) # Train a network
    plotGraph(trainLosses, testLosses, trainAcc, testAcc)
    plotDecisionBoundary(trainX, trainY)
    plotDecisionBoundary(testX, testY)

