"""
File: softmaxWithCIFAR10.py
Language: Python 3.5.1
Author: Karan Jariwala( kkj1811@rit.edu )
Description: Using softmax classifier on CIFAR-10 dataset
"""

__author__ = 'Karan Jariwala'
import itertools
import os, argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

from FinalSrcCode import softmaxClassifier as sm


def loadData(listOfTrainFilePath, testFilePath):
    """
    Load training and testing data from different file and separate the train/test
    input sample and labels
    :param listOfTrainFilePath: List of training batch file path
    :param testFilePath:  List of testing file path
    :return: train/test input sample and labels
    """
    xTrain = []
    yTrain = []

    for i in range(len(listOfTrainFilePath)):
        data, labels = loadBatch(listOfTrainFilePath[i])
        xTrain = data if i == 0 else np.concatenate([xTrain, data], axis=0)
        yTrain = labels if i == 0 else np.concatenate([yTrain, labels], axis=0)

    xTest, yTest = loadBatch(testFilePath)
    yTest = np.array(yTest)
    yTrain = np.array(yTrain)
    return xTrain, yTrain, xTest, yTest

def loadBatch(filePath):
    """
    Read a single file and return its data and labels
    :param filePath: A file path
    :return: data and labels
    """
    dataDict = unpickle(filePath)
    return dataDict[b'data'], dataDict[b'labels']

def unpickle(file):
    """
    Read the data from file and return the dictionary
    :param file: A file path
    :return: dictionary containing information of data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def vectorToRGBMatrix(xTrain, xTest):
    """
    Convert a vector data in 3-D matrix
    :param xTrain: Training sample
    :param xTest: Testing sample
    :return: 3-D matrix of train and test sample
    """
    xTrain = np.dstack((xTrain[:, :1024], xTrain[:, 1024:2048], xTrain[:, 2048:])) / 255.
    xTrain = np.reshape(xTrain, [-1, 32, 32, 3])
    xTest = np.dstack((xTest[:, :1024], xTest[:, 1024:2048], xTest[:, 2048:])) / 255.
    xTest = np.reshape(xTest, [-1, 32, 32, 3])
    return xTrain, xTest

def normalizeData(xTrain, xTest):
    """
    Normalize a data to -1 to 1 range. Subtract the mean from sample and divide it
    with 255.
    :param xTrain: Training sample
    :param xTest: Testing sample
    :return: normalize training and testing sample
    """
    meanImage = np.mean(xTrain, axis=0)
    xTrainD = xTrain - meanImage
    xTestD = xTest - meanImage
    xTrainD = np.divide(xTrainD, 255.)
    xTestD = np.divide(xTestD, 255.)
    return xTrainD, xTestD

def displayCifarImages(data, labels, meta, row=3, col=10, scale=4.):
    """
    Display number of different image from every class in single plot.
    :param data: Training data
    :param labels: Training classes
    :param meta: A list of classes name
    :param row: Number of row in a plot
    :param col: Number of column in a plot
    :param scale: how big the image should be
    :return: None
    """
    idx = count = 0
    imgWidth = imgHeight = 32
    keys = [lb for lb in range(len(meta))]
    labelCount = dict.fromkeys(keys, 0)

    figWidth = imgWidth / 80 * row * scale
    figHeight = imgHeight / 80 * col * scale
    fig, axes = plt.subplots(row, col, figsize=(figHeight, figWidth))

    while count < row * col:
        if labelCount[labels[idx]] >= row:
            idx += 1
            continue

        r = labelCount[labels[idx]]
        c = labels[idx]
        axes[r][c].imshow(data[idx])
        axes[r][c].set_title('{}: {}'.format(labels[idx],
                                             meta[labels[idx]].decode("utf-8")))
        axes[r][c].axis('off')
        plt.tight_layout()
        labelCount[labels[idx]] += 1
        count += 1
        idx += 1
    # plt.savefig(filepath)
    plt.show()

def getConfusionMatrix(actualLabel, predictedLabel, numOfClass):
    """
    Calculate a confusion matrix from actual label and predicted label
    :param actualLabel: Actual or target label
    :param predictedLabel: Predicted label
    :param numOfClass: Number of labels in dataset
    :return: confusion matrix of numOfclass x numOfclass
    """
    confMtrx =[]
    for _ in range(numOfClass):
        confMtrx.append([])
        for _ in range(numOfClass):
            confMtrx[-1].append(0)

    for sampleNum in range(actualLabel.shape[0]):
        confMtrx[int(actualLabel[sampleNum])][int(predictedLabel[sampleNum])] += 1
    confMtrx = np.array(confMtrx)
    return confMtrx

def plotConfusionMatrix(s81, xTest, actualLabel, classes, normalize=False,
                        title='Confusion matrix', cmap=plt.cm.Blues):
    """
    It display a confusion matrix graph.
    :param s81: A softmax classifier model
    :param xTest: Testing sample/data
    :param actualLabel: Actual or target label
    :param classes: class or label as a string
    :param normalize: To normalize or not
    :param title: Title for plot
    :param cmap: color map
    :return: None
    """
    predY = s81.predict(xTest)
    predY = predY.reshape((-1, 1))
    confMtrx = getConfusionMatrix(actualLabel, predY, 10)
    if normalize:
        confMtrx = confMtrx.astype('float') / confMtrx.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(confMtrx)

    plt.imshow(confMtrx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confMtrx.max() / 2.
    for i, j in itertools.product(range(confMtrx.shape[0]), range(confMtrx.shape[1])):
        plt.text(j, i, format(confMtrx[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confMtrx[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    """
    Main method
    """
    print("Started...")
    path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # complete path of training file
    TRAIN_FILENAMES = [os.path.join(path, 'data_batch_' + str(i)) for i in range(1, 6)]
    TEST_FILENAME = os.path.join(path, 'test_batch') # complete path of testing file
    META_FILENAME = os.path.join(path, 'batches.meta') # complete path of meta file

    meta = unpickle(META_FILENAME)
    meta = meta[b'label_names']

    print("Loading images and displaying it...")
    xTrain, yTrain, xTest, yTest = loadData(TRAIN_FILENAMES, TEST_FILENAME)
    xTrainMtrxRGB, xTestMtrxRGB = vectorToRGBMatrix(xTrain, xTest)
    displayCifarImages(xTrainMtrxRGB, yTrain, meta, row=3, col=10, scale=4.)
    xTrain, xTest = normalizeData(xTrain, xTest)
    yTrain = yTrain.reshape((-1, 1))
    yTest = yTest.reshape((-1, 1))

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs", default=100,
                        type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learningrate", dest="learningRate", default=0.001,
                        type=float, help="Learning rate or step size")
    parser.add_argument("-bs", "--batchSize", dest="batchSize", default=200,
                        type=int, help="Number of sample in mini-batches")
    parser.add_argument("-r", "--regStrength", dest="regStrength", default=0.001,
                        type=float, help="L2 weight decay regularization lambda value")
    parser.add_argument("-m", "--momentum", dest="momentum", default=0.005,
                        type=float, help="A momentum value")

    args = parser.parse_args()

    print(
        "Epochs: {} | Learning Rate: {} | Batch Size: {} | Regularization Strength: {} | "
        "Momentum: {} |".format(
            args.epochs,
            args.learningRate,
            args.batchSize,
            args.regStrength,
            args.momentum
        ))

    epochs = int(args.epochs)
    learningRate = float(args.learningRate)
    batchSize = int(args.batchSize)
    regStrength = int(args.regStrength)
    momentum = int(args.momentum)

    sftmx = sm.Softmax(epochs=epochs, learningRate=learningRate, batchSize=batchSize,
                       regStrength=regStrength, momentum=momentum)
    trainLosses, testLosses, trainAcc, testAcc = sftmx.train(xTrain, yTrain, xTest, yTest)
    sm.plotGraph(trainLosses, testLosses, trainAcc, testAcc)
    plotConfusionMatrix(sftmx, xTest, yTest, "0123456789",
                        normalize=True, title='Normalized confusion matrix')
    plt.show()