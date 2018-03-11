#Rachel Goldstein

import numpy as np
from scipy import stats

#######################################################################################

#sigmoid function
def sigmoid(x):
    return(1/(1+np.exp(-x)))

#loss function
#\sum p log p + (1-p) log (1-p)
def likelihood(w,x,y):
    sum = 0
    index = 0;
    for i in x:
        p=sigmoid(np.dot(w.T,i))
        m = np.dot(y[index],np.log(p))
        n = np.subtract(1,y[index])
        o = np.subtract(1,p)
        n = np.dot(n,np.log(o+.0000000001))
        m = np.add(m,n)
        #m+=n
        m=float(m)
        sum+=m
        index+=1
    return(sum)

#######################################################################################

#probability that output is 1 --> classification
def prob1(w,x,y):
    index = 0
    errors = 0
    #print(w.shape) shape = 58x1
    #print(y)
    for i in x:
        prob = sigmoid(np.dot(w.T,i)) #prob of y=1
        if prob>.5:
            if float(y[index])!=1:
                errors+=1
        else:
            if float(y[index])!=0:
                errors+=1
        index+=1
    return(errors/(index+1))

#######################################################################################

#gradient descent: (returns w)

def gradient(x,y,alpha):
    w_0 = np.random.normal(0,1,(58,1)) #w shape is 58 X 1
    i = y - sigmoid(np.dot(x,w_0))
    j = np.dot(alpha,x.T)
    w_1 = (w_0 + np.dot(j,i))
    
    print("initial likelihood is",likelihood(w_1,x,y))
    print("initial likelihood 2",likelihood(w_0,x,y))
    while np.absolute(likelihood(w_1,x,y) - likelihood(w_0,x,y)) >= 0.001:
        print("likelihood:",likelihood(w_1,x,y))
        w_0 = w_1
        
        i = y - sigmoid(np.dot(x,w_0))
        j = np.dot(alpha,x.T)
        w_1 = (w_0 + np.dot(j,i))
    
    
    return w_1




#######################################################################################
###########
#PARSE DATA FROM GIVEN FILE

with open('spam_train.txt', 'r') as f:
    spam_train_init=[[float(val) for val in line.split(',')[:58]] for line in f]

spamTrain = np.asarray(spam_train_init)
yTrain = spamTrain[:,57]
yTrain = yTrain.reshape(3065,1)

with open('spam_test.txt', 'r') as f:
    spam_test_init=[[float(val) for val in line.split(',')[:58]] for line in f]

spamTest = np.asarray(spam_test_init)
yTest = spamTest[:,57]
yTest = yTest.reshape(1536,1)

######################
#STANDARDIZE COLUMNS so all have mean 0 and unit variance
#USE Z SCORE FUNCTION

zSpamTrain = stats.zscore(spamTrain)
zSpamTest = stats.zscore(spamTest)

######################
#TRANSFORM FEATURES using log(x[ij] + 0.1)
trainTemp = np.ones(spamTrain.shape)
testTemp = np.ones(spamTest.shape)

trainTemp = np.multiply(0.1,trainTemp)
testTemp = np.multiply(0.1,testTemp) #matrices of desired shape, all elements 0.1

trainTemp = np.add(trainTemp,spamTrain)
testTemp = np.add(testTemp,spamTest)

#FINAL LOG MATRICES:
logSpamTrain = np.log(trainTemp)
logSpamTest = np.log(testTemp)

######################
#binarize features using I(x[ij] > 0)
#for elements f in matrix, if f > 0, f=1; if f<=0, f=0

binSpamTrain = np.where(spamTrain>0,1,0)
binSpamTest = np.where(spamTest>0,1,0)

#print("binSpamtrain:",binSpamTrain)

######################

#make last column of x matrices = 1:
newZspamTrain = np.matrix.copy(zSpamTrain)
newZspamTrain[:,57]=1
#print(yzTrain) #SOMEHOW THIS HAS MADE YZTRAIN ALL 1S?!?!

newZspamTest = np.matrix.copy(zSpamTest)
newZspamTest[:,57]=1

newlogSpamTrain = np.matrix.copy(logSpamTrain)
newlogSpamTrain[:,57]=1

newlogSpamTest = np.matrix.copy(logSpamTest)
newlogSpamTest[:,57]=1

newbinSpamTrain = np.matrix.copy(binSpamTrain)
newbinSpamTrain[:,57]=1

newbinSpamTest = np.matrix.copy(binSpamTest)
newbinSpamTest[:,57]=1


##################################################################
#LOGISTIC REGRESSION MODEL USING GRADIENT DESCENT

w_z = gradient(newZspamTrain,yTrain,0.001)
print("w_z")
w_log = gradient(newlogSpamTrain,yTrain,0.00001)
print("w_log")
w_bin = gradient(newbinSpamTrain,yTrain,0.001)
print("w_bin") #this is the only one actually entering gradient descent

##################################################################
#COMPUTE PROBABILITY

#REPORT EAN ERROR RATE ON TRAINING AND TEST SETS
#error rate = # of wrong classifications/total # of classifications

#probability on z training:
#print(yzTrain)
zTrainerror = prob1(w_z,newZspamTrain,yTrain)
print("error rate for z train:",zTrainerror)
print("error rate for z test:",prob1(w_z,newZspamTest,yTest))
print("error rate for log train:",prob1(w_log,newlogSpamTrain,yTrain))
print("error rate for log test:",prob1(w_log,newlogSpamTest,yTest))
print("error rate for bin train:",prob1(w_bin,newbinSpamTrain,yTrain))
print("error rate for bin test:",prob1(w_bin,newbinSpamTest,yTest))







