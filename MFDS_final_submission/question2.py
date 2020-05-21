
# importing the required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
np.random.seed(4)
# In[2]:
def standardize(df, label):                      #function to standardize the columns
    df=df.copy(deep=True)
    series=df.loc[:,label]
    avg=series.mean()
    stdv=series.std()
    series_standardized=(series-avg)/stdv
    return(series_standardized)

# creation of the data matrix
X_data = pd.read_csv("q2_data_matrix.csv",header = -1)
scaler = StandardScaler()
X_data = np.array(X_data)
X_data[:,0:4] = scaler.fit_transform(X_data[:,0:4])
Y_data = pd.read_csv("q2_labels.csv",header = -1)
Y_data = np.array(Y_data)
Data = np.concatenate((X_data,Y_data),axis = 1)
np.random.shuffle(Data)
X_data = Data[:,0:5]
Y_data = Data[:,5]
X_train = np.array(X_data[:700])
Y_train = np.array(Y_data[:700])
Y_train = Y_train.ravel()
X_valid = np.array(X_data[700:])
Y_valid = np.array(Y_data[700:])
Y_valid = Y_valid.ravel()

# In[3]:


def confusion_Matrix(Y_train,Y_pred):
    C = np.zeros((2,2))
    C[0,0] = int(np.sum((Y_train==0) & (Y_pred[:,None] == 0)))
    C[0,1] = int(np.sum((Y_train==0) & (Y_pred[:,None] == 1)))
    C[1,0] = int(np.sum((Y_train==1) & (Y_pred[:,None] == 0)))
    C[1,1] = int(np.sum((Y_train==1) & (Y_pred[:,None] == 1)))
    return(C)
            

# problem 
# using linear kernel
svclassifier = SVC(C=1,kernel = 'linear')
svclassifier.fit(X_train, Y_train)
Y_pred_tr = svclassifier.predict(X_train)
Y_pred = svclassifier.predict(X_valid)
print("Linear Kernel")
print("confusion matrix on train data")
print(confusion_matrix(Y_train,Y_pred_tr))
print("F1 score on train data")
print(classification_report(Y_train,Y_pred_tr))
print("confusion matrix on validation data")
print(confusion_matrix(Y_valid,Y_pred))
print("F1 score on validation data")
print(classification_report(Y_valid,Y_pred))
acc = np.count_nonzero(((Y_train-Y_pred_tr)==0))/700
print("Train accuracy :",acc)
acc = np.count_nonzero(((Y_valid-Y_pred)==0))/300
print("Validation accuracy :",acc)

# using rbf kernel
svclassifier = SVC(kernel = 'rbf')
svclassifier.fit(X_train, Y_train)
Y_pred_tr = svclassifier.predict(X_train)
Y_pred = svclassifier.predict(X_valid)
print("RBF Kernel")
print("confusion matrix on train data")
print(confusion_matrix(Y_train,Y_pred_tr))
print("F1 score on train data")
print(classification_report(Y_train,Y_pred_tr))
print("confusion matrix on validation data")
print(confusion_matrix(Y_valid,Y_pred))
print("F1 score on validation data")
print(classification_report(Y_valid,Y_pred))
acc = np.count_nonzero(((Y_train-Y_pred_tr)==0))/700
print("Train accuracy :",acc)
acc = np.count_nonzero(((Y_valid-Y_pred)==0))/300
print("Validation accuracy :",acc)

# using poly kernel
X_data = pd.read_csv("q2_data_matrix.csv",header = -1)
X_train = np.array(X_data[:700])
Y_train = np.array(Y_data[:700])
Y_train = Y_train.ravel()
X_valid = np.array(X_data[700:])
Y_valid = np.array(Y_data[700:])
Y_valid = Y_valid.ravel()
svclassifier = SVC(kernel = 'poly',gamma = 1e-8)
svclassifier.fit(X_train, Y_train)
Y_pred_tr = svclassifier.predict(X_train)
Y_pred = svclassifier.predict(X_valid)
print("Polynomial Kernel")
print("confusion matrix on train data")
print(confusion_matrix(Y_train,Y_pred_tr))
print("F1 score on train data")
print(classification_report(Y_train,Y_pred_tr))
print("confusion matrix on validation data")
print(confusion_matrix(Y_valid,Y_pred))
print("F1 score on validation data")
print(classification_report(Y_valid,Y_pred))
acc = np.count_nonzero(((Y_train-Y_pred_tr)==0))/700
print("Train accuracy :",acc)
acc = np.count_nonzero(((Y_valid-Y_pred)==0))/300
print("Validation accuracy :",acc)
