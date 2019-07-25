# -*- coding: utf-8 -*-

#  DO NOT CHANGE THIS PART!!!
#  DO NOT USE PACKAGES EXCEPT FOR THE PACKAGES THAT ARE IMPORTED BELOW!!!
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1AoCh22pmLHhdQtYdYUAJJqOCwF9obgVO', sep='\t')
data['class']=(data['class']=='g')*1

X=data.drop('class',axis=1).values
y=data['class'].values

trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)

#TODO: Logistic regression
#TODO: B - Predict output based on probability of class g (y=1)

clf = LogisticRegression()
clf.fit(trainX,trainY)
prob = clf.predict_proba(testX)[:,1]



#TODO: Draw a line plot (x=cutoff, y=accuracy)

a = []
acclist = []
for i in range(18):
    a += [round(00.1+i*0.05,2)]
for j in a:
    predict = []
    predict += [1 if i > j else 0 for i in prob ]
    #accuraccy = 올바르게 예측된 샘플/전체 샘플
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    acclist += [num/len(testY)]
plt.plot(a,acclist)


#TODO: Bernoulli naïve Bayes
#TODO: B - Estimate parameters of Bernoulli naïve Bayes
# Write user-defined function to estimate parameters of Bernoulli naïve Bayes
def BNB(X,y):
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    
    # initialize list

#TODO: calculate p values of several Bernoulli distributions    
    pmatrix=[]       
    
    trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)
   
    sample_mean=trainX.mean(0)
    n1 = 0
    n2 = 0
    for i in range(len(trainY)):
            if(trainY[i]==1):
                n1+=1
        
            elif(trainY[i]==0 ):
                n2+=1
    
    
    classification=1*(trainX>sample_mean)
    for k in range(10):#변수 1~10
        num1 = 0
        num2 = 0
        for i in range(len(trainY)):
            if(trainY[i]==1 and classification[i][k]==1):#클래스가 g일때
                num1 += 1
        
            elif(trainY[i]==0 and classification[i][k]==1):#클래스가 h일때
                num2 += 1
        result_g = num1/n1
        result_h = num2/n2
        
        pmatrix += [[result_g,result_h]]
        
    
    return pmatrix


#TODO: C - Predict output based on probability of class g (y=1)
#TODO: Draw a line plot (x=cutoff, y=accuracy)


pmatrix=BNB(X,y)
    
trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)
   
sample_mean=testX.mean(0)
   
classification=1*(testX>sample_mean)

n = 0
n2 =0
for i in testY:
    if i==1:
        n+=1
    else:
        n2+=1    

prob = []
plist = []
for i in classification:
    temp = 1
    temp2 = 1
    for j in range(len(i)):
        if(i[j]==1):
            temp *= pmatrix[j][0] #feature =1 class = g
            temp2 *= pmatrix[j][1] #feature =1 class = h
        else:
            temp *= (1-pmatrix[j][0])
            temp2 *= (1-pmatrix[j][1])
            
    plist.append((n*temp)/((n*temp) + (n2*temp2)))

a = []
acclist = []
for i in range(18):
    a += [round(00.1+i*0.05,2)]
for cutoff in a:
    predict = []
    predict += [1 if i > cutoff else 0 for i in plist ]
    
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    acclist.append(num /len(testY))
    
plt.plot(a,acclist)


#TODO: Nearest neighbor 
#TODO: B - k-NN with uniform weights
# Write user-deinfed function of k-NN 
# Use imported distance functions implemented by sklearn
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b
    
    # TODO: Euclidean distance
    d = 0
    d = euclidean_distances(a,b)
    return d

def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    
    # TODO: Manhattan distance
    d = 0
    d = manhattan_distances(a,b)
    return d

def knn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification
    y_pred = []

    dlist = np.array(dist(testX,trainX))
    
    for i in dlist:
        index_list = []
        class_list = []
        ddict = dict(zip(range(len(i)), i))   
        ddict = sorted(ddict, key=lambda o : ddict[o])
        index_list = ddict[0:k]
        for p in index_list:
            class_list.append(trainY[p])
        num=0
        for p in class_list:
            if p==1:
                num+=1
        if num >= (k+1)/2 :
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return y_pred  


# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using k-NN

#유클리디안

for i in range(3):
    predict = knn(trainX,trainY,testX,3+i*2,dist=euclidean_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))

#맨하탄

for i in range(3):
    predict = knn(trainX,trainY,testX,3+i*2,dist=manhattan_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))



  
#TODO: C - weighted k-NN
# Write user-deinfed function of weighted k-NN
# Use imported distance functions implemented by sklearn
def wknn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## Weighted K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: weighted k-NN classification
    y_pred = []

    euclidean_distances(testX,trainX)
    dlist = np.array(dist(testX,trainX))
    
    for i in dlist:
        index_list = []
        class_list = []
        weight_list = []
        cw_list = [0,0]
        ddict = dict(zip(range(len(i)), i))   
        ddict = sorted(ddict, key=lambda o : ddict[o])
        index_list = ddict[0:k]
        for p in index_list:
            class_list.append(trainY[p])
            weight_list.append(1/i[p])
        for t in range(k):
            if class_list[t] ==1:
                cw_list[1] += weight_list[t]
            else: cw_list[0] += weight_list[t]
        if cw_list[0]>cw_list[1]:
            y_pred.append(0)
        else: y_pred.append(1)
        
            
    return y_pred    



# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using weighted k-NN
    

#유클리디안

for i in range(3):
    predict = wknn(trainX,trainY,testX,3+i*2,dist=euclidean_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))

#맨하탄

for i in range(3):
    predict = wknn(trainX,trainY,testX,3+i*2,dist=manhattan_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))

#problem 3-D
#knn
#유클리디안

s_trainX = (trainX-trainX.mean(0))/trainX.std(0)
s_testX = (testX-trainX.mean(0))/testX.std(0)

for i in range(3):
    predict = knn(s_trainX,trainY,s_testX,3+i*2,dist=euclidean_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))


#맨하탄
s_trainX = (trainX-trainX.mean(0))/trainX.std(0)
s_testX = (testX-trainX.mean(0))/testX.std(0)
for i in range(3):
    predict = knn(s_trainX,trainY,s_testX,3+i*2,dist=manhattan_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))

#wknn
#유클리디안


s_trainX = (trainX-trainX.mean(0))/trainX.std(0)
s_testX = (testX-trainX.mean(0))/testX.std(0)

for i in range(3):
    predict = wknn(s_trainX,trainY,s_testX,3+i*2,dist=euclidean_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))

#맨하탄

s_trainX = (trainX-trainX.mean(0))/trainX.std(0)
s_testX = (testX-trainX.mean(0))/testX.std(0)

for i in range(3):
    predict = wknn(s_trainX,trainY,s_testX,3+i*2,dist=manhattan_dist)
    num =0
    for k in range(len(predict)):
        if predict[k] == testY[k]:
            num +=1
    print(num /len(testY))




#TODO: k-means clustering
#TODO: B - k-means clustering
# Write user-defined function of k-means clustering
def kmeans(X,k,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    
    # TODO: k-means clustering
    label = []
    centers = []
    #변수중에 고르기
    for i in np.random.randint(0,len(X),k):
        centers.append(X[i])
    centers = np.array(centers)
    for limit in range(max_iter):
        label = []#초기화
        
        dist = euclidean_dist(X,centers)
        for i in range(len(X)):
            tmp = []
            for j in range(len(centers)):
                tmp.append(dist[i][j])
            label.append(tmp.index((min(tmp))))
        
        old_centers = centers
        centers = np.zeros((k,len(X[0])))
        
        for i in range(len(label)):
            centers[label[i]] += X[i]
        unique, counts = np.unique(label, return_counts=True)
        for i in range(len(centers)):
            centers[i] /= counts[i]
            
        sum_distance = 0
        
        for a, b in zip(old_centers, centers):
            sum_distance +=  np.sqrt(np.sum((a-b)**2))
        if(sum_distance/k < 0.000001):
            break;
        
    return (label, centers)


# TODO: Calculate centroids of two clusters
    

label, centers = kmeans(X,2)

#TODO: C - homogeneity and completeness 

unique, counts = np.unique(y, return_counts=True)
class_list =[counts[0],counts[1]]
c0 = counts[0]
c1 = counts[1]

unique, counts = np.unique(label, return_counts=True)
kluster_list = [counts[0],counts[1]]
k0 = counts[0]
k1 = counts[1]
        
n_00 = 0
n_01 = 0
n_10 = 0
n_11 = 0

for i in range(len(X)):
    if y[i] == 0 and label[i] == 0:
        n_00 += 1
    elif y[i] == 0 and label[i] == 1:
        n_01 += 1
    elif y[i] == 1 and label[i] == 0:
        n_10 += 1
    elif y[i] == 1 and label[i] == 1:
        n_11 += 1
h_C=0
h_k=0
for i in range(2):
    h_C = -class_list[i]/len(X)*np.log2(class_list[i]/len(X))
    h_k = -kluster_list[i]/len(X)*np.log2(kluster_list[i]/len(X))

h_CK = -(n_00/len(X)*np.log2(n_00/k0) + n_01/len(X)*np.log2(n_01/k1) + n_10/len(X)*np.log2(n_10/k0) + n_11/len(X)*np.log2(n_11/k1))

h_KC = -(n_00/len(X)*np.log2(n_00/c0) + n_01/len(X)*np.log2(n_01/c0) + n_10/len(X)*np.log2(n_10/c1) + n_11/len(X)*np.log2(n_11/c1))

homogeneity = 1 - h_CK/h_C
completeness = 1 - h_KC/h_k
