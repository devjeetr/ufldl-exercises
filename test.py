import numpy as np
from SimpleNN import SimpleNN
import random
from math import floor
import struct, os
import gzip
from array import array
from numpy import zeros, uint8
from time import time
from sys import stdout
from Softmax import Softmax
from Autoencoder import AutoencoderNN

def init():
    print "hi"    

def Autoencoder():
    x_, y_ = getData("training_images.gz", "training_labels.gz")
    
    
    N = 60000
    
    x = x_[0:N].reshape(N, 784).T/255.0
    y = np.zeros((10, N))

    for i in xrange(N):
        y [y_[i][0]][i] = 1

    
    #nn1 = SimpleNN(784, 800, 10, 100, 0.15, 0.4, False)
    #nn2 = SimpleNN(784, 800, 10, 1, 0.15, 0.4, False)
    nn3 = AutoencoderNN(784, 50, 784, 10, 0.15, 0.05, 0.3, 0, False)
    nn4 = AutoencoderNN(784, 50, 784, 10, 0.35, 0.05, 0.3, 0, False)
    
    nn3.Train(x, x)
    nn4.Train(x, x)
    
    N = 10000    
    return
    x_, y_ = getData("test_images.gz", "test_labels.gz")
    x = x_.reshape(N, 784).T/255.0
    y = y_.T

    correct = np.zeros((4, 1))

    print "Testing"
    startTime = time()
    for i in xrange(N):
        #h1 = nn1.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        #h2 = nn2.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h3 = nn3.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h4 = nn4.Evaluate(np.tile(x.T[i].T, (1, 1)).T)

        #if h1[y[0][i]][0] > 0.8:
        #    correct[0][0] += 1

        #if h2[y[0][i]][0] > 0.8:
        #    correct[1][0] += 1

        if h3[y[0][i]][0] > 0.8:
            correct[2][0] += 1

        if h4[y[0][i]][0] > 0.8:
            correct[3][0] += 1

        if(i > 0):
            stdout.write("Testing %d/%d image. Time Elapsed: %ds. \r" % (i, N, time() - startTime))
            stdout.flush()

    stdout.write("\n")
    #print "Accuracy 1: ", correct[0][0]/10000.0 * 100, "%"
    #print "Accuracy 2: ", correct[1][0]/10000.0 * 100, "%"
    print "Accuracy 3: ", correct[2][0]/10000.0 * 100, "%"
    print "Accuracy 4: ", correct[3][0]/10000.0 * 100, "%" 


def testSoftmax():
    inputs, outputs = genXORData(12000)

    nn1 = Softmax(2, 21, 3, 2, 0.12, 0, False)

    nn1.Train(inputs, outputs)
    arr = np.zeros((2, 1))
    arr[0] = 1
    arr[1] = 1

    print "H\n",nn1.Evaluate(arr)
    arr[0] = 1
    arr[1] = 0
    print "R: OFF:\n",nn1.Evaluate(arr)
    arr[0] = 1
    arr[1] = 1
    print "H\n",nn1.Evaluate(arr)

def testSoftmaxMNIST():
    x_, y_ = getData("training_images.gz", "training_labels.gz")
    
    
    N = 600
    
    x = x_[0:N].reshape(N, 784).T/255.0
    y = np.zeros((10, N))

    for i in xrange(N):
        y [y_[i][0]][i] = 1

    
    #nn1 = SimpleNN(784, 800, 10, 100, 0.15, 0.4, False)
    #nn2 = SimpleNN(784, 800, 10, 1, 0.15, 0.4, False)
    nn3 = Softmax(784, 800, 1, 10, 0.15, 0, False)
    nn4 = Softmax(784, 800, 10, 10, 0.35, 0, False)
    
    #nn1.Train(x, y)
    #nn2.Train(x, y)
    nn3.Train(x, y)
    nn4.Train(x, y)
    
    N = 10000    
    
    x_, y_ = getData("test_images.gz", "test_labels.gz")
    x = x_.reshape(N, 784).T/255.0
    y = y_.T

    correct = np.zeros((4, 1))

    print "Testing"
    startTime = time()
    for i in xrange(N):
        #h1 = nn1.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        #h2 = nn2.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h3 = nn3.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h4 = nn4.Evaluate(np.tile(x.T[i].T, (1, 1)).T)

        #if h1[y[0][i]][0] > 0.8:
        #    correct[0][0] += 1

        #if h2[y[0][i]][0] > 0.8:
        #    correct[1][0] += 1

        if h3[y[0][i]][0] > 0.8:
            correct[2][0] += 1

        if h4[y[0][i]][0] > 0.8:
            correct[3][0] += 1

        if(i > 0):
            stdout.write("Testing %d/%d image. Time Elapsed: %ds. \r" % (i, N, time() - startTime))
            stdout.flush()

    stdout.write("\n")
    #print "Accuracy 1: ", correct[0][0]/10000.0 * 100, "%"
    #print "Accuracy 2: ", correct[1][0]/10000.0 * 100, "%"
    print "Accuracy 3: ", correct[2][0]/10000.0 * 100, "%"
    print "Accuracy 4: ", correct[3][0]/10000.0 * 100, "%"     

def testWithMNIST():
    x_, y_ = getData("training_images.gz", "training_labels.gz")
    
    
    N = 60000
    
    x = x_[0:N].reshape(N, 784).T/255.0
    y = np.zeros((10, N))

    for i in xrange(N):
        y [y_[i][0]][i] = 1

    
    #nn1 = SimpleNN(784, 800, 10, 100, 0.15, 0.4, False)
    #nn2 = SimpleNN(784, 800, 10, 1, 0.15, 0.4, False)
    nn3 = SimpleNN(784, 800, 10, 1, 0.15, 0, False)
    nn4 = SimpleNN(784, 1600, 10, 1, 0.15, 0, False)
    
    #nn1.Train(x, y)
    #nn2.Train(x, y)
    nn3.Train(x, y)
    nn4.Train(x, y)
    
    N = 10000    
    
    x_, y_ = getData("test_images.gz", "test_labels.gz")
    x = x_.reshape(N, 784).T/255.0
    y = y_.T

    correct = np.zeros((4, 1))

    print "Testing"
    startTime = time()
    for i in xrange(N):
        #h1 = nn1.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        #h2 = nn2.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h3 = nn3.Evaluate(np.tile(x.T[i].T, (1, 1)).T)
        h4 = nn4.Evaluate(np.tile(x.T[i].T, (1, 1)).T)

        #if h1[y[0][i]][0] > 0.8:
        #    correct[0][0] += 1

        #if h2[y[0][i]][0] > 0.8:
        #    correct[1][0] += 1

        if h3[y[0][i]][0] > 0.8:
            correct[2][0] += 1

        if h4[y[0][i]][0] > 0.8:
            correct[3][0] += 1

        if(i > 0):
            stdout.write("Testing %d/%d image. Time Elapsed: %ds. \r" % (i, N, time() - startTime))
            stdout.flush()

    stdout.write("\n")
    #print "Accuracy 1: ", correct[0][0]/10000.0 * 100, "%"
    #print "Accuracy 2: ", correct[1][0]/10000.0 * 100, "%"
    print "Accuracy 3: ", correct[2][0]/10000.0 * 100, "%"
    print "Accuracy 4: ", correct[3][0]/10000.0 * 100, "%"      


def testWithXOR():
    inputs, outputs = genXORData(12000)

    nn1 = SimpleNN(2, 21, 3, 10, 0.3, 0.1, False)
    nn2 = SimpleNN(2, 21, 3, 10, 0.3, 0.4, False)

    nn1.Train(inputs, outputs)
    nn2.Train(inputs, outputs)
    arr = np.zeros((2, 1))
    arr[0] = 1
    arr[1] = 1
    print "R: OFF:\n",nn1.Evaluate(arr), "\nR: ON\n", nn2.Evaluate(arr)
    arr[0] = 1
    arr[1] = 0
    print "R: OFF:\n",nn1.Evaluate(arr), "\nR: ON\n", nn2.Evaluate(arr)
    arr[0] = 0
    arr[1] = 1
    print "R: OFF:\n",nn1.Evaluate(arr), "\nR: ON\n", nn2.Evaluate(arr)

def genXORData(size):
    inputs = np.zeros((2, size))
    outputs = np.zeros((3, size))

    for i in xrange(size):
        inputs[0][i] = floor((random.random() * 100)) % 2  # 0 or 1
        inputs[1][i] = floor((random.random() * 100)) % 2  # 0 or 1
        
        outputs[0][i] = 0
        outputs[1][i] = 0
        outputs[2][i] = 0

        if inputs[0][i] == inputs[1][i]:
            outputs[0][i] = 1

        if inputs[0][i] > inputs[1][i]:
            outputs[1][i] = 1

        if inputs[0][i] < inputs[1][i]:
            outputs[2][i] = 1


    #print "1:", freq1, ", 2:", freq2
    return inputs, outputs






def getData(imagePath, labelPath):

    imageFile, labelFile = gzip.open(os.path.join(".", imagePath), 'rb'), gzip.open(os.path.join(".", labelPath), 'rb')

    iMagic, iSize, rows, cols = struct.unpack('>IIII', imageFile.read(16))
    lMagic, lSize = struct.unpack('>II', labelFile.read(8))

    x = zeros((lSize, rows, cols), dtype=uint8)
    y = zeros((lSize, 1), dtype=uint8)
    count = 0

    startTime = time()

    for i in range(lSize):
        for row in range(rows):
            for col in range(cols):
                x[i][row][col] = struct.unpack(">B", imageFile.read(1))[0]

        y[i] = struct.unpack(">B", labelFile.read(1))[0]
        count = count + 1
        if count % 101 == 0:
            stdout.write("Image: %d/%d. Time Elapsed: %ds  \r" % (i, lSize, time() - startTime))
            stdout.flush()
        #if count > 600:
#            break
    stdout.write("\n")

    return (x, y)




