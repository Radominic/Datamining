# -*- coding: utf-8 -*-

# Do not change this part
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 

# Data load
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1O74eCM8zlPxCFEuEpshmFXA3HpKT1qbC')

#TODO: explanatory analysis

#TODO: B, C, D - Darw pairwise scatter plots
#max_temperatur - Mean_temperature
X = data[data.columns[0]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[-1])
#Min_temperature - Mean_temperature
X = data[data.columns[1]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[1])
plt.ylabel(data.columns[-1])
#Dewpoint - Mean_temperature
X = data[data.columns[2]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[2])
plt.ylabel(data.columns[-1])
#Precipitation - Mean_temperature
X = data[data.columns[3]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[3])
plt.ylabel(data.columns[-1])
#Sea_level_pressure - Mean_temperature
X = data[data.columns[4]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[4])
plt.ylabel(data.columns[-1])
#Standard_pressure - Mean_temperature
X = data[data.columns[5]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[5])
plt.ylabel(data.columns[-1])
#Visibility - Mean_temperature
X = data[data.columns[6]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[6])
plt.ylabel(data.columns[-1])
#Wind_speed - Mean_temperature
X = data[data.columns[7]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[7])
plt.ylabel(data.columns[-1])
#Max_wind_speed - Mean_temperature
X = data[data.columns[8]]
Y = data[data.columns[-1]]
plt.scatter(X,Y)
plt.xlabel(data.columns[8])
plt.ylabel(data.columns[-1])

#TODO: E, F - Calculate correlation matrix
cor_mat = data.corr()
cor_mat

plt.matshow(data.corr())
plt.xticks(range(len(data.columns)), data.columns)
plt.xticks()
plt.yticks(range(len(data.columns)), data.columns)
plt.colorbar()
plt.show()

#TODO: linear regression

#TODO: B - Calculate VIF
reg = LinearRegression()
VIF = []
for i in range(0,9):
    clist = []
    for j in range(0,9):
        if j!=i:
            clist.append(data.columns[j])
    reg.fit(data[clist],data[data.columns[i]])
    VIF.append(1/(1-reg.score(data[clist],data[data.columns[i]])))

 

# Model 1
#TODO: C - Train a lineare regression model
#          Calculate t-test statistics 

#datalist
target_list = []
for i in range(len(VIF)):
    if VIF[i] < 5:
        target_list.append(data.columns[i])
reg = LinearRegression()
reg.fit(data[target_list],data['Mean_temperature'])

y_pred = reg.predict(data[target_list])
SSE = sum((data['Mean_temperature']-y_pred)**2)
n = len(data)
p = 4
MSE = SSE/(n-p-1)

X = data[target_list].values
X = np.c_[np.ones(n),X]
xtx = np.matmul(X.T,X)
xtx_inv = np.linalg.inv(xtx)

se_list = []
for i in range(p+1):
    se_list.append(np.sqrt(MSE*xtx_inv[i,i]))

beta = np.insert(reg.coef_,0,reg.intercept_)

t = beta/se_list

ci = np.append(reg.intercept_,reg.coef_)

pvalue = 1-tdist.cdf(np.abs(t),n-p-1)

data_table = {'coef' : ci ,
            'standard error': se_list,
            't-statistics': t,
            'p-values': pvalue}

data_table = pd.DataFrame(data_table)

#TODO: D - Calculate adjusted R^2

r_square = reg.score(data[target_list],data[data.columns[-1]])
temp1 = (1-r_square)
temp2 = (n-1)/(n-p-1)
adj_r_square = 1-temp1*temp2

#TODO: E - Apply F-test

SSR = sum((y_pred - data[data.columns[-1]].mean())**2)
MSR = SSR/p

F = MSR/MSE
F_test = 1-fdist.cdf(F,p,n-p-1)

# Model 2
#TODO: F - Check the appropriateness of the given set of variables

new_list = ['Visibility','Dewpoint','Precipitation']
reg = LinearRegression()
VIF = []
for i in range(0,3):
    clist = []
    for j in range(0,3):
        if j!=i:
            clist.append(new_list[j])
    reg.fit(data[clist],data[new_list[i]])
    VIF.append(1/(1-reg.score(data[clist],data[new_list[i]])))


#TODO: G - Train a lineare regression model
#          Calculate t-test statistics 

reg = LinearRegression()
reg.fit(data[new_list],data['Mean_temperature'])

y_pred = reg.predict(data[new_list])
SSE = sum((data['Mean_temperature']-y_pred)**2)
n = len(data)
p = 3
MSE = SSE/(n-p-1)

X = data[new_list].values
X = np.c_[np.ones(n),X]
xtx = np.matmul(X.T,X)
xtx_inv = np.linalg.inv(xtx)

se_list = []
for i in range(p+1):
    se_list.append(np.sqrt(MSE*xtx_inv[i,i]))

beta = np.insert(reg.coef_,0,reg.intercept_)

t = beta/se_list

ci = np.append(reg.intercept_,reg.coef_)

pvalue = 1-tdist.cdf(np.abs(t),n-p-1)

data_table = {'coef' : ci ,
            'standard error': se_list,
            't-statistics': t,
            'p-values': pvalue}

data_table = pd.DataFrame(data_table)


#TODO: H - Calculate adjusted R^2

r_square = reg.score(data[new_list],data[data.columns[-1]])
temp1 = (1-r_square)
temp2 = (n-1)/(n-p-1)
adj_r_square = 1-temp1*temp2


#TODO: I - Test normality of residuals

e = data[data.columns[-1]] - y_pred


S = ((1/n)*(sum((e-e.mean())**3))) / (((1/n)*(sum((e-e.mean())**2)))**(3/2))
C = (1/n)*(sum((e-e.mean())**4)) / ((1/n)*(sum((e-e.mean())**2)))**(4/2)

JB = ((n-p)/6)*(S**2+(1/4)*((C-3)**2))

JB_test = 1-chi2.cdf(JB,2)

#TODO: J - Test homoskedasticty of residuals

LM = n*reg.score(data[new_list],e**2)
LM
LM_test = 1-chi2.cdf(LM,p)