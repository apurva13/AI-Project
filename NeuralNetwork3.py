import sys
import math as m
import pandas as pd
import numpy as np
import time

groupLength = 32
learningRate = 0.01
epoch = 100
inputSize = 784
outputSize = 10

class DigitIdentification:

    def activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))
        # return 1.0 / (1.0 + np.exp(-(np.clip(x, -500, 500))))

    def weights_bias(self,lval,uval,sval):
        wbVal = np.random.randn(lval, uval) * np.sqrt(1 / sval)
        return wbVal

    def __init__(self, learningRate, groupLength, epoch):
        self.W1 = self.weights_bias(256,784,784)
        self.W2 = self.weights_bias(64,256,256)
        self.W3 = self.weights_bias(10,64,64)
        self.b1 = self.weights_bias(256,1,784)
        self.b2 = self.weights_bias(64,1,256)
        self.b3 = self.weights_bias(10,1,64)
        # self.W1 = np.random.randn(256, 784) * np.sqrt(1 / 784)
        # self.W2 = np.random.randn(64, 256) * np.sqrt(1 / 256)
        # self.W3 = np.random.randn(10, 64) * np.sqrt(1 / 64)
        # self.b1 = np.random.randn(256, 1) * np.sqrt(1 / 784)
        # self.b2 = np.random.randn(64, 1) * np.sqrt(1 / 256)
        # self.b3 = np.random.randn(10, 1) * np.sqrt(1 / 64)
        self.learningRate = learningRate
        self.groupLength = groupLength
        self.epoch = epoch


    def onehot_Y(self, trainLabel_Y):
        onehotY = np.zeros((trainLabel_Y.size, outputSize))
        onehotY[np.arange(trainLabel_Y.size), trainLabel_Y] = 1
        onehotY = onehotY.T
        return onehotY

    def activationDerivative(self, dA, x):
        return dA * self.activation(x) * (1 - self.activation(x))

    def softmax(self, x):
        return np.exp(x - x.max()) / np.sum(np.exp(x - x.max()), axis=0)

    def cross_entropy_loss(self, Y, pred_Y):
        lossValue = (-1 / Y.shape[1]) * np.sum(np.multiply(Y, np.log(pred_Y)))
        lossValue = np.squeeze(lossValue)  # remove single-dimensional values from the shape of an array.
        return lossValue

    def get_accuracy(self, X, Y):
        predictions = []
        cache = self.feedForward(X)
        # pred = np.argmax(cache, axis=0)
        predictions.append(np.argmax(cache, axis=0) == np.argmax(Y, axis=0))
        return np.mean(predictions)

    # Chunking the Data into Batches
    def shortBatches(self, y_train, X_train, groupLength):
        batchesValue = list()
        tVal = X_train.shape[1]
        group = m.floor(X_train.shape[1] / groupLength)

        # Creating the valid chunks for X and y
        for i in range(0, group):
            validY = y_train[:, i * groupLength: (i + 1) * groupLength]
            validX = X_train[:, i * groupLength: (i + 1) * groupLength]
            batchesValue.append((validX, validY))

        if tVal % groupLength != 0:
            valCount = groupLength * m.floor(tVal / groupLength)
            validY = y_train[:, valCount: tVal]
            validX = X_train[:, valCount: tVal]

            sBatch = (validX, validY)
            batchesValue.append(sBatch)

        return batchesValue

    def feedForward(self, X):

        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.activation(self.Z2)
        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    # backward pass
    def backwardPropogation(self, X, Y):
        tVal = X.shape[1]
        errorZ3 = self.A3 - Y
        self.dW3 = np.dot(errorZ3, self.A2.T) / tVal
        self.db3 = np.sum(errorZ3, axis=1, keepdims=True) / tVal
        self.dA2 = np.dot(self.W3.T, errorZ3)
        self.dZ2 = self.activationDerivative(self.dA2, self.Z2)
        self.dW2 = np.dot(self.dZ2, self.A1.T) / tVal
        self.db2 = np.sum(self.dZ2, axis=1, keepdims=True) / tVal
        self.dA1 = np.dot(self.W2.T, self.dZ2)
        self.dZ1 = self.activationDerivative(self.dA1, self.Z1)
        self.dW1 = np.dot(self.dZ1, X.T) / tVal
        self.db1 = np.sum(self.dZ1, axis=1, keepdims=True) / tVal

    # Defining the Training
    def training(self, X_train, y_train):
        for i in range(0, self.epoch):

            indexValue = np.arange(len(X_train[1]))
            np.random.shuffle(indexValue)

            loss = 0

            X_shuffle = X_train[:, np.arange(len(X_train[1]))]

            valLength = np.arange(len(X_train[1]))

            Y_shuffle = y_train[:, np.arange(len(X_train[1]))]
            batchesValue = self.shortBatches(Y_shuffle, X_shuffle, self.groupLength)

            # Initialing the batch values
            for sBatch in batchesValue:
                # Setting the Valid X and y on the basis of Batch Size
                validX, validY = sBatch
                cal = self.feedForward(validX)

                # Calling the Backward Propogation function
                self.backwardPropogation(validX, validY)
                loss += self.cross_entropy_loss(validY, cal)

                # Defining the value of W and b
                self.W1 = self.W1 - (self.learningRate * self.dW1)
                self.b1 = self.b1 - (self.learningRate * self.db1)
                self.W2 = self.W2 - (self.learningRate * self.dW2)
                self.b2 = self.b2 - (self.learningRate * self.db2)
                self.W3 = self.W3 - (self.learningRate * self.dW3)
                self.b3 = self.b3 - (self.learningRate * self.db3)

            val = self.get_accuracy(X_shuffle, Y_shuffle)
            print("Epoch:", i, " Accuracy:", val * 100)


start = time.time()
# Read Input the data
if (len(sys.argv)>1):
    trainImageInput = pd.read_csv(sys.argv[1], header=None)
    trainLabelInput = pd.read_csv(sys.argv[2], header=None)
    testImageInput = pd.read_csv(sys.argv[3], header=None)
    testLabelInput = pd.read_csv(sys.argv[4], header=None)
else:
    print("-work directory input-")
    trainImageInput = pd.read_csv("train_image.csv", header=None)
    trainLabelInput = pd.read_csv("train_label.csv", header=None)
    testImageInput = pd.read_csv("test_image.csv", header=None)
    testLabelInput = pd.read_csv("test_label.csv", header=None)

rows,col= trainImageInput.shape
set = 10000

if rows<10000:
    set=rows

trainImageInput = trainImageInput[0:set]

# Transpose and convert deframe to numpy arrays
trainImage_X = (trainImageInput.T).values
trainLabel_Y = (trainLabelInput.T).values
testImage_X = (testImageInput.T).values
testLabel_Y = (testLabelInput.T).values

mlPerpetron = DigitIdentification(learningRate, groupLength, epoch)

onehotY = mlPerpetron.onehot_Y(trainLabel_Y)

# train DigitIdentification
mlPerpetron.training(trainImage_X, onehotY)

cache = mlPerpetron.feedForward(testImage_X)
output = mlPerpetron.A3
pred = np.argmax(output, axis=0)

# write out the test set predictions
pd.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
print(time.time()-start)