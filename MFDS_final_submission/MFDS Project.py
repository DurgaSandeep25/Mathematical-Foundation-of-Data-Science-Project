#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
data = pd.read_csv("F:\I.I.T.M\SEMESTER\Sem6\MFDS\Project\IPL_Twitter_MissingData.csv")
#x = 1000-data.count(axis=0)
print(data.isnull().sum())


# In[ ]:


# drop rows with missing values
data.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(data.shape)
# fill missing values with mean column values
#data.fillna(data.mean(), inplace=True)
# count the number of NaN values in each column
print(data.isnull().sum())


# In[ ]:


data.head()


# In[ ]:


#Extracting specific columns of a pandas dataframe
Z = data[["X1", "X2", "X3", "X4"]]
Zs = Z - Z.mean()
covariance = Zs.cov()
#To convert a pandas dataframe (df) to a numpy ndarray
covariance = covariance.values
from numpy import linalg as LA
lamda, v = LA.eig(covariance)
min_index = np.argmin(lamda, axis = 0)
#Linear relationship among them are given by least eigen value
print(v[:,min_index])
print(data.shape)


# In[ ]:


data_b = data[(data['Q1']==0)  & (data['Q2']==0)]
print(data_b)


# In[ ]:


#Extracting specific columns of a pandas dataframe
Z_b = data_b[["X1", "X2", "X3", "X4"]]
Zs_b = Z_b - Z_b.mean()
covariance_b = Zs_b.cov()
#To convert a pandas dataframe (df) to a numpy ndarray
covariance_b = covariance_b.values
lamda_b, v_b = LA.eig(covariance_b)
min_index_b = np.argmin(lamda_b, axis = 0)
#Linear relationship among them are given by least eigen value
print(v_b[:,min_index_b])
print(data_b.shape)


# In[ ]:


data_c = data[(data['Q1']==0)  & (data['Q2']==1)]
print(data_c.shape)


# In[ ]:


#Extracting specific columns of a pandas dataframe
Z_c = data_c[["X1", "X2", "X3", "X4"]]
Zs_c = Z_c - Z_c.mean()
covariance_c = Zs_c.cov()
#To convert a pandas dataframe (df) to a numpy ndarray
covariance_c = covariance_c.values
lamda_c, v_c= LA.eig(covariance_c)
min_index_c = np.argmin(lamda_c, axis = 0)
#Linear relationship among them are given by least eigen value
print(v_c[:,min_index_c])
print(data_c.shape)


# In[ ]:


data_d = data[(data['Q1']==1)  & (data['Q2']==0)]
print(data_d.shape)
#Extracting specific columns of a pandas dataframe
Z_d = data_d[["X1", "X2", "X3", "X4"]]
Zs_d = Z_d - Z_d.mean()
covariance_d = Zs_d.cov()
#To convert a pandas dataframe (df) to a numpy ndarray
covariance_d = covariance_d.values
lamda_d, v_d= LA.eig(covariance_d)
min_index_d = np.argmin(lamda_d, axis = 0)
#Linear relationship among them are given by least eigen value
print(v_d[:,min_index_d])
print(data_d.shape)


# In[ ]:


data_e = data[(data['Q1']==1)  & (data['Q2']==1)]
print(data_e.shape)
#Extracting specific columns of a pandas dataframe
Z_e= data_e[["X1", "X2", "X3", "X4"]]
Zs_e= Z_e- Z_e.mean()
covariance_e= Zs_e.cov()
#To convert a pandas dataframe (df) to a numpy ndarray
covariance_e= covariance_e.values
lamda_e, v_e= LA.eig(covariance_e)
min_index_e = np.argmin(lamda_e, axis = 0)
#Linear relationship among them are given by least eigen value
print(v_e[:,min_index_e])
print(data_e.shape)


# In[ ]:


#Impute the missing values


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = pd.read_csv("F:\I.I.T.M\SEMESTER\Sem6\MFDS\Project\q2_data_matrix.csv")
y = pd.read_csv("F:\I.I.T.M\SEMESTER\Sem6\MFDS\Project\q2_labels.csv")
y = y.values
y = y.ravel()


# In[2]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)


# In[4]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[8]:


# Kernel linear, polynomial, gaussian
svclassifier1 = SVC(kernel = 'rbf')
svclassifier1.fit(X_train, y_train)
y_pred1 = svclassifier1.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


# In[ ]:




