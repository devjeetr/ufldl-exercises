import numpy as np

class SimpleNN:

    def __init__(self, nI=2, nH=21, nO=1, batchSize=10, learningRate=0.3, decayRate=0.3, verbose=False):
        self.nI = nI
        self.nH = nH
        self.nO = nO

        self.batchSize = batchSize
        self.learningRate = learningRate
        self.decayRate = decayRate
        self.verbose = verbose

        self.w1 = np.random.normal(0, 0.4, (nH, nI))
        self.w2 = np.random.normal(0, 0.4, (nO, nH))

        self.b1 = np.random.normal(0, 0.6, (nH, 1))
        self.b2 = np.random.normal(0, 0.6, (nO, 1))

    def Train(self, inputs, labels):
        N = labels.shape[1]
        print N
        for i in xrange(0, N, self.batchSize):
            y  = labels.T[i:i + self.batchSize].T

            x  = inputs.T[i:i + self.batchSize].T
            z2 = self.w1.dot(x) + np.tile(self.b1, (1, self.batchSize))
            a2 = self.__Sigmoid(z2)
            z3 = self.w2.dot(a2) + np.tile(self.b2, (1, self.batchSize))
            h  = self.__Sigmoid(z3)


            delta3 = h - y
            
            delta2 = self.w2.T.dot(delta3) * self.__SigmoidPrime(z2)

            gradW2 = delta3.dot(a2.T)/self.batchSize
            gradW1 = delta2.dot(x.T)/self.batchSize

            eps = 0.0004
            gradTable = np.zeros(gradW2.shape)

            #cost = self.__computeCost(y, a2, self.w2, self.b2, self.batchSize)
            #print "cost:", cost

            #if abs(cost) < 0.001 and i/self.batchSize > 2:
                #print "cost of 0.001 achieved in", i/self.batchSize, "epochs"
                #break

            self.w2 = self.w2 - self.learningRate * (gradW2 - self.decayRate/self.batchSize * self.w2)
            self.w1 = self.w1 - self.learningRate * (gradW1 - self.decayRate/self.batchSize * self.w1)

            gradB2 = np.tile(np.sum(delta3, axis=1), (1, 1)).T
            gradB1 = np.tile(np.sum(delta2, axis=1), (1, 1)).T

            #rint ">>>>>", gradB2.shape, gradB1.shape

            self.b2 -= self.learningRate * gradB2 
            self.b1 -= self.learningRate * gradB1


            
    def Evaluate(self, x):
        
        z2 = self.w1.dot(x) + self.b1
        a2 = self.__Sigmoid(z2)
        z3 = self.w2.dot(a2)
        h = self.__Sigmoid(z3)

        return h


    def __computeGradientsNumerically():
        print 

    def __computeCost(self, y, x, theta, b, m):
        h = self.__Sigmoid(theta.dot(x) + np.tile(b, (1, self.batchSize)))
        cost = -1/m * (np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + self.decayRate/(2 * m) * np.sum(theta**2))
        return cost

    def __Sigmoid(self, x):
        return 1/(1 + np.exp(-1 * x))

    def __SigmoidPrime(self, x):
        return self.__Sigmoid(x) * (1 - self.__Sigmoid(x))    